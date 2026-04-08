use super::decoder::decode_ctc;
use crate::ocr::config::RecognizerConfig;
use crate::ocr::runtime::{REC_MEAN, REC_NORM, build_net};
use crate::ocr::types::TextLine;
use anyhow::{Context, Result, bail};
use image::RgbImage;
use ncnnrs::{Mat, MatPixelType, Net, Option as NcnnOption};
use std::fs;
use std::path::Path;

/// 文本识别器。
///
/// 负责加载字典和识别模型，并将裁剪后的文本块解码为字符串。
pub(crate) struct Recognizer {
    net: Net,
    opt: NcnnOption,
    dict: Vec<String>,
    cfg: RecognizerConfig,
}

impl Recognizer {
    /// 根据识别模型和字典文件创建识别器。
    pub(crate) fn new(
        model_stem: impl AsRef<Path>,
        dict_path: impl AsRef<Path>,
        num_threads: u32,
        gpu_index: Option<u32>,
        cfg: RecognizerConfig,
    ) -> Result<Self> {
        let (net, opt) = build_net(model_stem, num_threads, gpu_index)?;
        let dict_text = fs::read_to_string(dict_path.as_ref())
            .with_context(|| format!("failed to read dict {}", dict_path.as_ref().display()))?;
        let dict = dict_text.lines().map(|line| line.to_owned()).collect();

        Ok(Self {
            net,
            opt,
            dict,
            cfg,
        })
    }

    /// 对单个裁剪文本块执行识别。
    pub(crate) fn recognize(&self, image: &RgbImage) -> Result<TextLine> {
        let scale = self.cfg.dst_height as f32 / image.height().max(1) as f32;
        let dst_width = ((image.width() as f32 * scale).round() as u32).max(1);
        let mut input = Mat::from_pixels_resize(
            image.as_raw(),
            MatPixelType::RGB,
            image.width() as i32,
            image.height() as i32,
            dst_width as i32,
            self.cfg.dst_height as i32,
            None,
        )?;
        input.substract_mean_normalize(&REC_MEAN, &REC_NORM);

        let mut extractor = self.net.create_extractor();
        extractor.set_option(&self.opt);
        extractor.input("in0", &input)?;
        let mut out = Mat::new();
        extractor.extract("out0", &mut out)?;

        let steps = out.h() as usize;
        let classes = out.w() as usize;
        if classes != self.dict.len() + 1 {
            bail!(
                "dict/model mismatch: rec output classes={} but dict entries={} (expected classes=dict+1). try PaddleOCR's ppocrv5_dict.txt for PP_OCRv5_mobile_rec",
                classes,
                self.dict.len()
            );
        }

        let logits = out.data_as_slice::<f32>(steps * classes);
        decode_ctc(logits, steps, classes, &self.dict)
    }
}
