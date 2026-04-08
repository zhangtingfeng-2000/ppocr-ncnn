use anyhow::{bail, Result};
use image::{ImageReader, RgbImage};
use ppocr_ncnn::prelude::*;
use std::env;
use std::path::{Path, PathBuf};

pub const RUNS: usize = 10;

pub struct RuntimePaths {
    pub root: PathBuf,
    pub det_model: PathBuf,
    pub rec_model: PathBuf,
    pub dict_path: PathBuf,
}

pub struct MetricColumn {
    pub header: &'static str,
    pub accessor: fn(&OcrTiming) -> f64,
}

impl MetricColumn {
    pub const fn new(header: &'static str, accessor: fn(&OcrTiming) -> f64) -> Self {
        Self { header, accessor }
    }
}

pub fn resolve_runtime_paths() -> Result<RuntimePaths> {
    let root = find_runtime_root()?;
    Ok(RuntimePaths {
        det_model: root
            .join("models")
            .join("det")
            .join("PP_OCRv5_mobile_det.ncnn"),
        rec_model: root
            .join("models")
            .join("rec")
            .join("PP_OCRv5_mobile_rec.ncnn"),
        dict_path: root.join("models").join("rec").join("ppocrv5_dict.txt"),
        root,
    })
}

pub fn load_input_image(runtime_root: &Path, image_name: &str) -> Result<RgbImage> {
    let image_path = runtime_root.join("inputs").join(image_name);
    Ok(ImageReader::open(&image_path)?.decode()?.into_rgb8())
}

pub fn print_benchmark(
    engine: &OcrEngine,
    image: &RgbImage,
    runs: usize,
    metrics: &[MetricColumn],
) -> Result<()> {
    let warmup = engine.detect_image(image)?;
    let recognized_text = warmup.text.clone();
    let warmup_headers = benchmark_headers("stage", metrics);
    let warmup_rows = vec![measurement_row(
        "warmup",
        &warmup.timing,
        warmup.blocks.len(),
        metrics,
    )];
    print_ascii_table("warmup", &warmup_headers, &warmup_rows);
    println!();

    let mut metric_values: Vec<Vec<f64>> = (0..metrics.len())
        .map(|_| Vec::with_capacity(runs))
        .collect();
    let mut rows = Vec::with_capacity(runs);

    for run in 0..runs {
        let result = engine.detect_image(image)?;
        for (index, metric) in metrics.iter().enumerate() {
            metric_values[index].push((metric.accessor)(&result.timing));
        }
        rows.push(measurement_row(
            &format!("{:02}", run + 1),
            &result.timing,
            result.blocks.len(),
            metrics,
        ));
    }

    let run_headers = benchmark_headers("run", metrics);
    print_ascii_table("runs", &run_headers, &rows);
    println!();

    let summary_rows: Vec<Vec<String>> = metrics
        .iter()
        .zip(metric_values.iter())
        .map(|(metric, values)| summary_row(metric.header.trim_end_matches("_ms"), values))
        .collect();
    print_ascii_table(
        "summary",
        &["metric", "runs", "min_ms", "max_ms", "avg_ms"],
        &summary_rows,
    );

    if !recognized_text.is_empty() {
        println!();
        println!("text");
        println!("{recognized_text}");
    }

    Ok(())
}

pub fn print_ascii_table(title: &str, headers: &[&str], rows: &[Vec<String>]) {
    let mut widths: Vec<usize> = headers.iter().map(|header| header.len()).collect();

    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if index >= widths.len() {
                widths.push(cell.len());
            } else {
                widths[index] = widths[index].max(cell.len());
            }
        }
    }

    println!("{title}");
    print_border(&widths);
    print_row(headers.iter().copied(), &widths);
    print_border(&widths);

    for row in rows {
        print_row(row.iter().map(String::as_str), &widths);
    }

    print_border(&widths);
}

fn benchmark_headers<'a>(first: &'a str, metrics: &'a [MetricColumn]) -> Vec<&'a str> {
    let mut headers = Vec::with_capacity(metrics.len() + 2);
    headers.push(first);
    headers.extend(metrics.iter().map(|metric| metric.header));
    headers.push("blocks");
    headers
}

fn measurement_row(
    label: &str,
    timing: &OcrTiming,
    blocks: usize,
    metrics: &[MetricColumn],
) -> Vec<String> {
    let mut row = Vec::with_capacity(metrics.len() + 2);
    row.push(label.to_string());
    row.extend(
        metrics
            .iter()
            .map(|metric| format!("{:.2}", (metric.accessor)(timing))),
    );
    row.push(blocks.to_string());
    row
}

fn print_border(widths: &[usize]) {
    print!("+");
    for width in widths {
        print!("{:-<1$}+", "", width + 2);
    }
    println!();
}

fn print_row<'a>(cells: impl IntoIterator<Item = &'a str>, widths: &[usize]) {
    print!("|");
    for (index, cell) in cells.into_iter().enumerate() {
        print!(" {:<width$} |", cell, width = widths[index]);
    }
    println!();
}

pub fn summary_row(label: &str, values: &[f64]) -> Vec<String> {
    let min = values
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    let avg = values.iter().copied().sum::<f64>() / values.len() as f64;

    vec![
        label.to_string(),
        values.len().to_string(),
        format!("{min:.2}"),
        format!("{max:.2}"),
        format!("{avg:.2}"),
    ]
}

fn find_runtime_root() -> Result<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd);
    }

    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.to_path_buf());
        candidates.extend(dir.ancestors().skip(1).take(4).map(PathBuf::from));
    }

    if let Some(workspace_root) = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent() {
        candidates.push(workspace_root.to_path_buf());
    }

    for candidate in candidates {
        if has_runtime_layout(&candidate) {
            return Ok(candidate);
        }
    }

    bail!("failed to locate runtime root containing models/ and inputs/")
}

fn has_runtime_layout(path: &Path) -> bool {
    path.join("models").is_dir() && path.join("inputs").is_dir()
}

pub fn total_ms(timing: &OcrTiming) -> f64 {
    timing.total_ms
}

pub fn det_ms(timing: &OcrTiming) -> f64 {
    timing.detect_ms
}

pub fn prep_ms(timing: &OcrTiming) -> f64 {
    timing.prepare_ms
}

pub fn crop_ms(timing: &OcrTiming) -> f64 {
    timing.crop_ms
}

#[allow(dead_code)]
pub fn cls_ms(timing: &OcrTiming) -> f64 {
    timing.angle_cls_ms
}

pub fn rec_ms(timing: &OcrTiming) -> f64 {
    timing.recognize_ms
}

pub fn post_ms(timing: &OcrTiming) -> f64 {
    timing.postprocess_ms
}
