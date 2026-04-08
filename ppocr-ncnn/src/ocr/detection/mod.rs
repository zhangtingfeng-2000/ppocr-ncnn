//! 文本检测子模块。
//!
//! 包含检测推理和文本框后处理逻辑。

mod inference;
mod postprocess;

pub(crate) use inference::Detector;
