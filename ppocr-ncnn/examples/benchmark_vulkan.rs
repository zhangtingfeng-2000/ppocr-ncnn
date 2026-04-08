use anyhow::{bail, Result};
use ncnnrs::{get_device_name, get_gpu_count};
use ppocr_ncnn::prelude::*;

mod common;

use common::{
    cls_ms, crop_ms, det_ms, load_input_image, post_ms, prep_ms, print_benchmark, rec_ms,
    resolve_runtime_paths, total_ms, MetricColumn, RUNS,
};

const METRICS: &[MetricColumn] = &[
    MetricColumn::new("total_ms", total_ms),
    MetricColumn::new("prep_ms", prep_ms),
    MetricColumn::new("det_ms", det_ms),
    MetricColumn::new("crop_ms", crop_ms),
    MetricColumn::new("cls_ms", cls_ms),
    MetricColumn::new("rec_ms", rec_ms),
    MetricColumn::new("post_ms", post_ms),
];
const GPU_INDEX: u32 = 0;

fn main() -> Result<()> {
    let gpu_count = get_gpu_count();
    if gpu_count <= 0 {
        bail!("no Vulkan-capable ncnn device available");
    }
    if GPU_INDEX as i32 >= gpu_count {
        bail!(
            "requested gpu_index={} but only {} device(s) are available",
            GPU_INDEX,
            gpu_count
        );
    }

    let runtime = resolve_runtime_paths()?;

    let config = OcrConfig {
        gpu_index: Some(GPU_INDEX),
        ..Default::default()
    };

    println!(
        "using Vulkan gpu_index={} device={}",
        GPU_INDEX,
        get_device_name(GPU_INDEX as i32)
    );

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
