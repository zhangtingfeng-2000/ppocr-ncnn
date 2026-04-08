use anyhow::Result;
use ncnnrs::{get_device_name, get_gpu_count};
use ppocr_ncnn::prelude::*;

mod common;

use common::{
    crop_ms, det_ms, load_input_image, post_ms, prep_ms, print_benchmark, rec_ms, resolve_runtime_paths,
    total_ms, MetricColumn, RUNS,
};

const METRICS: &[MetricColumn] = &[
    MetricColumn::new("total_ms", total_ms),
    MetricColumn::new("prep_ms", prep_ms),
    MetricColumn::new("det_ms", det_ms),
    MetricColumn::new("crop_ms", crop_ms),
    MetricColumn::new("rec_ms", rec_ms),
    MetricColumn::new("post_ms", post_ms),
];
const GPU_INDEX: u32 = 0;

fn main() -> Result<()> {
    let runtime = resolve_runtime_paths()?;

    let cpu_config = OcrConfig::default();
    let cpu_engine = OcrEngine::new(
        &runtime.det_model,
        &runtime.rec_model,
        &runtime.dict_path,
        None::<&str>,
        cpu_config,
    )?;

    let vulkan_engine = if get_gpu_count() > GPU_INDEX as i32 {
        let vulkan_config = OcrConfig {
            gpu_index: Some(GPU_INDEX),
            ..Default::default()
        };
        Some((
            get_device_name(GPU_INDEX as i32).to_owned(),
            OcrEngine::new(
                &runtime.det_model,
                &runtime.rec_model,
                &runtime.dict_path,
                None::<&str>,
                vulkan_config,
            )?,
        ))
    } else {
        None
    };

    println!("runtime root: {}", runtime.root.display());
    println!("cpu backend: enabled");
    match &vulkan_engine {
        Some((device_name, _)) => println!(
            "vulkan backend: enabled gpu_index={} device={}",
            GPU_INDEX, device_name
        ),
        None => println!("vulkan backend: unavailable, skipping Vulkan benchmarks"),
    }
    println!();

    for image_name in ["1.png", "2.png"] {
        let image = load_input_image(&runtime.root, image_name)?;

        run_backend("cpu", image_name, &cpu_engine, &image)?;

        if let Some((_, engine)) = &vulkan_engine {
            println!();
            run_backend("vulkan", image_name, engine, &image)?;
        }

        println!();
    }

    Ok(())
}

fn run_backend(
    backend: &str,
    image_name: &str,
    engine: &OcrEngine,
    image: &image::RgbImage,
) -> Result<()> {
    println!("=== {backend} / {image_name} ===");
    print_benchmark(engine, image, RUNS, METRICS)?;
    Ok(())
}
