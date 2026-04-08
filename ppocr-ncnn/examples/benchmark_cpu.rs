use anyhow::Result;
use ppocr_ncnn::prelude::*;

mod common;

use common::{
    MetricColumn, RUNS, crop_ms, det_ms, load_input_image, post_ms, prep_ms, print_benchmark,
    rec_ms, resolve_runtime_paths, total_ms,
};

const METRICS: &[MetricColumn] = &[
    MetricColumn::new("total_ms", total_ms),
    MetricColumn::new("prep_ms", prep_ms),
    MetricColumn::new("det_ms", det_ms),
    MetricColumn::new("crop_ms", crop_ms),
    MetricColumn::new("rec_ms", rec_ms),
    MetricColumn::new("post_ms", post_ms),
];

fn main() -> Result<()> {
    let runtime = resolve_runtime_paths()?;

    let config = OcrConfig::default();
    let engine = OcrEngine::new(
        &runtime.det_model,
        &runtime.rec_model,
        &runtime.dict_path,
        None::<&str>,
        config,
    )?;
    let image = load_input_image(&runtime.root, "1.png")?;
    print_benchmark(&engine, &image, RUNS, METRICS)?;

    Ok(())
}

