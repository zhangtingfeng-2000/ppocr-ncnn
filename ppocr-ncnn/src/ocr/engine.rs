use super::classification::AngleClassifier;
use super::config::OcrConfig;
use super::detection::Detector;
use super::recognition::Recognizer;
use super::types::{OcrResult, OcrTiming, Quad, TextBlock};
use crate::geometry::order_quad;
use crate::image_ops::{crop_quad, draw_quad, make_padding, rotate_180};
use anyhow::{Context, Result};
use image::{ImageReader, Rgb, RgbImage};
use std::path::Path;
use std::time::Instant;

/// OCR 引擎，负责串联文本检测、可选角度分类和文本识别整个流程。
pub struct OcrEngine {
    detector: Detector,
    recognizer: Recognizer,
    angle_classifier: Option<AngleClassifier>,
    config: OcrConfig,
}

impl OcrEngine {
    /// 根据检测、识别和可选角度分类模型构建 OCR 引擎。
    ///
    /// `det_model`、`rec_model` 和 `cls_model` 既可以传入完整的
    /// `.param` / `.bin` 路径，也可以传入共享前缀，例如 `model.ncnn`。
    pub fn new(
        det_model: impl AsRef<Path>,
        rec_model: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        cls_model: Option<impl AsRef<Path>>,
        config: OcrConfig,
    ) -> Result<Self> {
        let detector = Detector::new(det_model, config.num_threads, config.gpu_index)?;
        let recognizer = Recognizer::new(
            rec_model,
            dict_path,
            config.num_threads,
            config.gpu_index,
            config.recognizer.clone(),
        )?;
        let angle_classifier = match cls_model {
            Some(path) => Some(AngleClassifier::new(
                path,
                config.num_threads,
                config.gpu_index,
                config.angle.clone(),
            )?),
            None => None,
        };

        Ok(Self {
            detector,
            recognizer,
            angle_classifier,
            config,
        })
    }

    /// 从磁盘加载图片并执行完整 OCR 流程。
    pub fn detect_path(&self, image_path: impl AsRef<Path>) -> Result<OcrResult> {
        let image_path = image_path.as_ref();
        let image = ImageReader::open(image_path)
            .with_context(|| format!("failed to open image {}", image_path.display()))?
            .decode()
            .with_context(|| format!("failed to decode image {}", image_path.display()))?
            .into_rgb8();
        self.detect_image(&image)
    }

    /// 对内存中的 RGB 图片执行 OCR。
    ///
    /// 返回值包含检测后的文本块、拼接后的完整文本、可选的框图以及耗时信息。
    pub fn detect_image(&self, image: &RgbImage) -> Result<OcrResult> {
        let total_start = Instant::now();
        let padding = self.config.detector.padding;

        let prepare_start = Instant::now();
        let padded = make_padding(image, padding);
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

        let detect_start = Instant::now();
        let padded_boxes = self.detector.detect(&padded, &self.config.detector)?;
        let detect_ms = detect_start.elapsed().as_secs_f64() * 1000.0;

        let mut crop_ms = 0.0f64;
        let mut angle_cls_ms = 0.0f64;
        let mut recognize_ms = 0.0f64;
        let postprocess_start = Instant::now();
        let mut blocks = Vec::with_capacity(padded_boxes.len());

        for text_box in padded_boxes {
            let crop_start = Instant::now();
            let crop = crop_quad(&padded, &text_box.quad)?;
            let angle = if let Some(classifier) = &self.angle_classifier {
                let start = Instant::now();
                let angle = classifier.classify(&crop)?;
                angle_cls_ms += start.elapsed().as_secs_f64() * 1000.0;
                Some(angle)
            } else {
                None
            };
            let crop = match angle.as_ref().map(|value| value.index) {
                Some(0) => rotate_180(&crop),
                _ => crop,
            };
            crop_ms += crop_start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let text_line = self.recognizer.recognize(&crop)?;
            recognize_ms += start.elapsed().as_secs_f64() * 1000.0;
            let quad = remove_padding(text_box.quad, padding, image.width(), image.height());
            blocks.push(TextBlock {
                quad,
                box_score: text_box.score,
                angle,
                text: text_line.text,
                char_scores: text_line.char_scores,
            });
        }

        let text = blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let box_image = if self.config.draw_boxes {
            let mut annotated = image.clone();
            for block in &blocks {
                draw_quad(&mut annotated, &block.quad, Rgb([255, 0, 0]));
            }
            Some(annotated)
        } else {
            None
        };
        let postprocess_ms = postprocess_start.elapsed().as_secs_f64() * 1000.0
            - crop_ms
            - angle_cls_ms
            - recognize_ms;

        Ok(OcrResult {
            blocks,
            text,
            box_image,
            timing: OcrTiming {
                total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
                prepare_ms,
                detect_ms,
                crop_ms,
                angle_cls_ms,
                recognize_ms,
                postprocess_ms: postprocess_ms.max(0.0),
            },
        })
    }
}

fn remove_padding(quad: Quad, padding: u32, width: u32, height: u32) -> Quad {
    if padding == 0 {
        return order_quad(quad);
    }

    let padding = padding as f32;
    let mut out = quad;
    for point in &mut out {
        point.x = (point.x - padding).clamp(0.0, width.saturating_sub(1) as f32);
        point.y = (point.y - padding).clamp(0.0, height.saturating_sub(1) as f32);
    }
    order_quad(out)
}
