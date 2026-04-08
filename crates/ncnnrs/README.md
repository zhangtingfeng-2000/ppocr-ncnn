# ncnnrs

`ncnnrs` 是这个仓库里给 `ppocr-ncnn` 使用的 `ncnn` Rust绑定层。
修改自 [Baiyuetribe/ncnnrs](https://github.com/Baiyuetribe/ncnnrs)

它的实现分成两部分：

- 底层 FFI 绑定由 `build.rs` 在编译时通过 `bindgen` 从本地安装的 `ncnn` 头文件生成。
- 上层 `Net`、`Extractor`、`Mat`、`Option`、`Allocator` 等 Rust 类型，是围绕 `ncnn` C API 写的一层手工封装，目标是满足当前 OCR
  推理链路，而不是做一个完整的通用 SDK。

## 编译要求

按当前仓库实现，`ncnnrs` 的编译要求如下。

### 1. Rust 工具链

- 需要支持 `edition = "2024"` 的 Rust 工具链。
- 实际上建议使用较新的 stable 工具链。

### 2. `bindgen` 依赖

因为 `build.rs` 会在本地生成绑定，所以需要：

- 可用的 Clang / libclang 环境
- `bindgen` 能正常找到并解析 C/C++ 头文件

如果本机没有可用的 `libclang`，构建会在生成绑定阶段失败。

### 3. 本地安装的 `ncnn`

这个 crate 不会自动下载或编译 `ncnn`，需要你先准备好一份本地可用的 `ncnn` 头文件和库文件。

当前 `build.rs` 的硬性要求是：

- `NCNN_INCLUDE_DIR` 必须存在
- `NCNN_INCLUDE_DIR` 下必须能找到 `c_api.h`
- `NCNN_INCLUDE_DIR` 下还会读取 `gpu.h`
- 库目录里必须能找到 `ncnn.lib`
- 当前还会额外链接 `glslang.lib`

也就是说，至少要满足下面这种目录结构：

```text
<ncnn-root>\
  x64\
    include\
      ncnn\
        c_api.h
        gpu.h
    lib\
      ncnn.lib
      glslang.lib
```

ncnn可以从 https://github.com/Tencent/ncnn/releases 下载已编译好的库。

### 4. 环境变量

构建时会读取两个环境变量：

- `NCNN_INCLUDE_DIR`
    - 必填
    - 指向 `ncnn` 头文件目录，例如 `E:\env\ncnn-20260113-windows-vs2022\x64\include\ncnn`

- `NCNN_LIB_DIR`
    - 可选
    - 如果不设置，`build.rs` 会根据 `NCNN_INCLUDE_DIR` 自动推导到上两级目录下的 `lib`
    - 对上面的例子，会自动推导到 `E:\env\ncnn-20260113-windows-vs2022\x64\lib`

### 5. 当前平台约束

当前 `build.rs` 会直接检查 `ncnn.lib`，因此现状明显偏向 Windows/MSVC 的库布局。

这意味着：

- 当前实现已经按 Windows 静态库命名方式写死
- 还没有为 Linux/macOS 的 `.a` / `.so` / `.dylib` 做分支处理
- 如果要交叉编译其他平台，需要先调整 `build.rs`

### 6. `cpu` feature 的实际含义

`Cargo.toml` 里有一个 `cpu` feature：

```toml
[features]
cpu = []
```

这个 feature 当前只会影响 Rust 侧是否调用 GPU 相关运行时逻辑，不会改变 `build.rs` 的头文件解析和链接行为。

因此即使启用：

```powershell
cargo check -p ncnnrs --features cpu
```

当前版本依然需要：

- `gpu.h`
- `ncnn.lib`
- `glslang.lib`
