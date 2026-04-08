use crate::ocr::types::TextLine;
use anyhow::{Context, Result};

/// 对识别模型输出执行 CTC 贪心解码。
pub(super) fn decode_ctc(
    logits: &[f32],
    steps: usize,
    classes: usize,
    dict: &[String],
) -> Result<TextLine> {
    let mut text = String::new();
    let mut char_scores = Vec::new();
    let mut previous_index = 0usize;

    for step in 0..steps {
        let row = &logits[step * classes..(step + 1) * classes];
        let mut best_index = 0usize;
        let mut best_prob = 0.0f32;

        for (index, &value) in row.iter().enumerate() {
            if value > best_prob {
                best_prob = value;
                best_index = index;
            }
        }

        if best_index > 0 && best_index != previous_index {
            let dict_index = best_index - 1;
            let token = dict
                .get(dict_index)
                .with_context(|| format!("dict index {} out of range", dict_index))?;
            text.push_str(token);
            char_scores.push(best_prob);
        }
        previous_index = best_index;
    }

    Ok(TextLine { text, char_scores })
}
