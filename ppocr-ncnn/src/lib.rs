mod geometry;
mod image_ops;
pub mod ocr;

/// 常用公开 API 的预导出模块。
///
/// 使用方式：
/// `use ppocr_ncnn::prelude::*;`
pub mod prelude {
    pub use crate::ocr::{
        Angle, AngleConfig, DetectorConfig, OcrConfig, OcrEngine, OcrResult, OcrTiming, Point,
        Quad, RecognizerConfig, ScaleParam, TextBlock, TextBox, TextLine,
    };
}
