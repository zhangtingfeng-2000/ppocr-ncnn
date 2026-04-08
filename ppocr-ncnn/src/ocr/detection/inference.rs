use super::postprocess::{compute_scale_param, postprocess_boxes};
use crate::ocr::config::DetectorConfig;
use crate::ocr::runtime::{DET_MEAN, DET_NORM, build_net};
use crate::ocr::types::TextBox;
use anyhow::Result;
use image::RgbImage;
use ncnnrs::{Mat, MatPixelType, Net, Option as NcnnOption};
use std::path::Path;

/// 文本检测器。
///
/// 负责将输入图片送入检测模型，并把输出概率图交给后处理阶段。
pub(crate) struct Detector {
    net: Net,
    opt: NcnnOption,
}

impl Detector {
    /// 根据检测模型创建文本检测器。
    pub(crate) fn new(
        model_stem: impl AsRef<Path>,
        num_threads: u32,
        gpu_index: Option<u32>,
    ) -> Result<Self> {
        let (net, opt) = build_net(model_stem, num_threads, gpu_index)?;
        Ok(Self { net, opt })
    }

    /// 执行文本检测并返回排序后的候选文本框。
    pub(crate) fn detect(&self, image: &RgbImage, cfg: &DetectorConfig) -> Result<Vec<TextBox>> {
        let scale = compute_scale_param(image, cfg.max_side_len);
        let mut input = Mat::from_pixels_resize(
            image.as_raw(),
            MatPixelType::RGB,
            image.width() as i32,
            image.height() as i32,
            scale.dst_width as i32,
            scale.dst_height as i32,
            None,
        )?;
        input.substract_mean_normalize(&DET_MEAN, &DET_NORM);

        let mut extractor = self.net.create_extractor();
        extractor.set_option(&self.opt);
        extractor.input("in0", &input)?;
        let mut out = Mat::new();
        extractor.extract("out0", &mut out)?;
        let width = out.w() as usize;
        let height = out.h() as usize;
        let prob_map = out.data_as_slice::<f32>(width * height);
        postprocess_boxes(prob_map, width, height, &scale, cfg)
    }
}
