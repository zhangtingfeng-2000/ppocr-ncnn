//! OCR 模块入口。
//!
//! 该模块按职责拆分为配置、引擎、检测、识别、角度分类、运行时和公共类型。

mod classification;
mod config;
mod detection;
mod engine;
mod recognition;
mod runtime;
mod types;

pub use config::{AngleConfig, DetectorConfig, OcrConfig, RecognizerConfig};
pub use engine::OcrEngine;
pub use types::{
    Angle, OcrResult, OcrTiming, Point, Quad, ScaleParam, TextBlock, TextBox, TextLine,
};
