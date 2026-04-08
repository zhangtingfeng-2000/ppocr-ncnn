use anyhow::{Context, Result};
use ncnnrs::{Net, Option as NcnnOption};
use std::path::{Path, PathBuf};

/// 检测模型预处理均值。
pub(crate) const DET_MEAN: [f32; 3] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0];
/// 检测模型预处理归一化系数。
pub(crate) const DET_NORM: [f32; 3] = [
    1.0 / 0.229 / 255.0,
    1.0 / 0.224 / 255.0,
    1.0 / 0.225 / 255.0,
];
/// 识别与分类模型预处理均值。
pub(crate) const REC_MEAN: [f32; 3] = [127.5, 127.5, 127.5];
/// 识别与分类模型预处理归一化系数。
pub(crate) const REC_NORM: [f32; 3] = [1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5];

/// 创建并初始化一个 ncnn 网络及其运行选项。
pub(crate) fn build_net(
    model_stem: impl AsRef<Path>,
    num_threads: u32,
    gpu_index: Option<u32>,
) -> Result<(Net, NcnnOption)> {
    let mut net = Net::new();
    let mut opt = build_option(num_threads, gpu_index);
    opt.set_use_fp16_packed(true);
    opt.set_use_fp16_storage(false); // 存储不使用fp16
    opt.set_use_fp16_arithmetic(true);
    net.set_option(&opt);
    if let Some(index) = gpu_index {
        net.set_vulkan_device(index);
    }
    load_model_stem(&mut net, model_stem)?;
    Ok((net, opt))
}

/// 构建 ncnn 推理选项。
fn build_option(num_threads: u32, gpu_index: Option<u32>) -> NcnnOption {
    let mut opt = NcnnOption::new();
    opt.set_num_threads(num_threads);
    opt.use_vulkan_compute(gpu_index.is_some());
    opt
}

/// 根据模型前缀加载 `.param` 与 `.bin` 文件。
fn load_model_stem(net: &mut Net, model_stem: impl AsRef<Path>) -> Result<()> {
    let stem = model_stem.as_ref();
    let param = with_extension(stem, "param");
    let bin = with_extension(stem, "bin");
    net.load_param(param.to_str().context("non-utf8 param path")?)?;
    net.load_model(bin.to_str().context("non-utf8 model path")?)?;
    Ok(())
}

/// 为模型前缀补上指定扩展名。
fn with_extension(stem: &Path, ext: &str) -> PathBuf {
    if stem.extension().is_some_and(|value| value == ext) {
        stem.to_path_buf()
    } else {
        PathBuf::from(format!("{}.{}", stem.display(), ext))
    }
}
