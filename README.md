# ppocr-ncnn

一个基于 Rust + `ncnn` 的 PaddleOCR 推理实验仓库。

当前工作区包含两个 crate：

- `ppocr-ncnn`
    - OCR 业务层，负责文本检测、可选角度分类、文本识别、结果整理和 benchmark 输出
- `crates/ncnnrs`
    - `ncnn` 的 Rust FFI 和轻量封装，供 `ppocr-ncnn` 调用

仓库内已经放入一组运行示例所需的模型和测试图片，拿到仓库后可以直接跑 benchmark。

## 参考来源

- PaddleOCR
    - 本项目使用的是 PP-OCR 系列模型和字典文件
- Tencent ncnn
    - 推理后端基于 `ncnn` C API

`ncnnrs` 的更细节说明见 [crates/ncnnrs/README.md](./crates/ncnnrs/README.md)。

## 仓库结构

```text
.
├─ Cargo.toml
├─ crates/
│  └─ ncnnrs/
├─ ppocr-ncnn/
│  ├─ src/
│  └─ examples/
├─ models/
│  ├─ det/
│  ├─ rec/
│  └─ cls/
└─ inputs/
```

当前示例默认会在仓库根目录下查找：

- `models/det/PP_OCRv5_mobile_det.ncnn.param`
- `models/det/PP_OCRv5_mobile_det.ncnn.bin`
- `models/rec/PP_OCRv5_mobile_rec.ncnn.param`
- `models/rec/PP_OCRv5_mobile_rec.ncnn.bin`
- `models/rec/ppocrv5_dict.txt`
- `inputs/1.png`
- `inputs/2.png`

## 功能概览

当前实现已经串起以下 OCR 流程：

1. 图片预处理和补边
2. 文本检测
3. 透视裁剪
4. 可选角度分类
5. 文本识别
6. 文本块拼接和耗时统计

公开入口主要是 `ppocr_ncnn::prelude::*` 中导出的：

- `OcrEngine`
- `OcrConfig`
- `OcrResult`
- `TextBlock`
- `OcrTiming`

## 编译要求

### Rust

- 需要较新的 stable Rust
- 工作区使用 `edition = "2024"`

### bindgen / libclang

`ncnnrs` 会在编译时生成绑定，所以还需要：

- 可用的 Clang / libclang
- `bindgen` 能成功解析本地 `ncnn` 头文件

## 快速开始

### 1. 配置环境变量

```powershell
$env:NCNN_INCLUDE_DIR="path/to/ncnn"
```

按需补充：

```powershell
$env:NCNN_LIB_DIR="path/to/lib"
```

### 2. 检查项目能否编译

```powershell
cargo check
```

### 3. 运行示例(测试性能)

只跑 CPU benchmark：

```powershell
cargo run -p ppocr-ncnn --example cpu-only --release
```

只跑 Vulkan benchmark：

```powershell
cargo run -p ppocr-ncnn --example vulkan-only --release
```

同时跑 CPU 和 Vulkan benchmark：

```powershell
cargo run -p ppocr-ncnn --example all --release
```

## 示例代码

```rust
use ppocr_ncnn::prelude::*;

fn main() -> anyhow::Result<()> {
    let config = OcrConfig::default();
    let engine = OcrEngine::new(
        "models/det/PP_OCRv5_mobile_det.ncnn",
        "models/rec/PP_OCRv5_mobile_rec.ncnn",
        "models/rec/ppocrv5_dict.txt",
        None::<&str>,
        config,
    )?;

    let result = engine.detect_path("inputs/1.png")?;
    println!("{}", result.text);
    Ok(())
}
```

说明：

- `det_model` 和 `rec_model` 既可以传完整 `.param` / `.bin` 路径，也可以传共享前缀，例如 `xxx.ncnn`
- `cls_model` 是可选的；当前仓库里已有 `models/cls/`，但示例默认没有启用分类模型
- `gpu_index: None` 表示只在运行时禁用 GPU，不代表构建时不需要 GPU 相关头文件和库

## Feature 说明

`ppocr-ncnn` 目前只有一个 feature：

```toml
[features]
cpu = ["ncnnrs/cpu"]
```

它的作用是把底层 `ncnnrs` 切到 CPU 运行路径，因此：

- `cpu-only` 示例需要 `--features cpu`
- 开启 `cpu` feature 后，GPU 查询接口会退化为 CPU-only 行为
- 这个 feature 目前不会移除编译阶段对 `gpu.h` 和 `glslang.lib` 的依赖

## 当前限制

- 当前 `ncnnrs` 构建脚本明显偏向 Windows/MSVC，直接检查 `ncnn.lib`
- 还没有为 Linux/macOS 的 `.a` / `.so` / `.dylib` 做兼容分支
- `cpu` feature 目前不是“真正的 CPU-only 构建”
- 示例默认使用仓库内置的模型目录布局，如果调整目录结构，需要同步修改路径或自定义调用代码
