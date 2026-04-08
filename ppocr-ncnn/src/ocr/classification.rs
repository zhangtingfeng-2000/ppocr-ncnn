use super::config::AngleConfig;
use super::runtime::{REC_MEAN, REC_NORM, build_net};
use super::types::Angle;
use crate::image_ops::adjust_target_img;
use anyhow::Result;
use image::RgbImage;
use ncnnrs::{Mat, MatPixelType, Net, Option as NcnnOption};
use std::path::Path;

/// 文本角度分类器。
///
/// 用于判断裁剪后的文本块是否需要旋转，以提升识别稳定性。
pub(crate) struct AngleClassifier {
    net: Net,
    opt: NcnnOption,
    cfg: AngleConfig,
}

impl AngleClassifier {
    /// 根据分类模型创建角度分类器。
    pub(crate) fn new(
        model_stem: impl AsRef<Path>,
        num_threads: u32,
        gpu_index: Option<u32>,
        cfg: AngleConfig,
    ) -> Result<Self> {
        let (net, opt) = build_net(model_stem, num_threads, gpu_index)?;
        Ok(Self { net, opt, cfg })
    }

    /// 对单个文本块进行角度分类，返回最高分的类别与置信度。
    pub(crate) fn classify(&self, image: &RgbImage) -> Result<Angle> {
        let adjusted = adjust_target_img(image, self.cfg.dst_width, self.cfg.dst_height);
        let mut input = Mat::from_pixels(
            adjusted.as_raw(),
            MatPixelType::RGB,
            adjusted.width() as i32,
            adjusted.height() as i32,
            None,
        )?;
        input.substract_mean_normalize(&REC_MEAN, &REC_NORM);

        let mut extractor = self.net.create_extractor();
        extractor.set_option(&self.opt);
        extractor.input("input", &input)?;
        let mut out = Mat::new();
        extractor.extract("output", &mut out)?;

        let scores = out.data_as_slice::<f32>(out.w() as usize);
        let (index, score) = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((1, 0.0));
        Ok(Angle { index, score })
    }
}
