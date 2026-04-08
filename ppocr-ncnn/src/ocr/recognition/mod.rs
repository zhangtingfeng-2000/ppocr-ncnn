//! 文本识别子模块。
//!
//! 包含识别推理和 CTC 解码逻辑。

mod decoder;
mod inference;

pub(crate) use inference::Recognizer;
