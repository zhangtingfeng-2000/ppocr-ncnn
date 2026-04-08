/// 文本检测与检测后处理配置。
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// 检测前缩放时允许的最长边。
    pub max_side_len: u32,
    /// 候选文本框区域的平均置信度阈值。
    pub box_score_thresh: f32,
    /// 检测概率图二值化阈值。
    pub box_thresh: f32,
    /// 文本框外扩比例，用于在裁剪前适当放大候选区域。
    pub unclip_ratio: f32,
    /// 检测前在原图四周补白的像素值。
    pub padding: u32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            max_side_len: 960,
            box_score_thresh: 0.6,
            box_thresh: 0.3,
            unclip_ratio: 2.0,
            padding: 0,
        }
    }
}

/// 文本识别预处理配置。
#[derive(Debug, Clone)]
pub struct RecognizerConfig {
    /// 识别模型输入目标高度。
    pub dst_height: u32,
}

impl Default for RecognizerConfig {
    fn default() -> Self {
        Self { dst_height: 32 }
    }
}

/// 可选角度分类模型配置。
#[derive(Debug, Clone)]
pub struct AngleConfig {
    /// 角度分类器输入目标宽度。
    pub dst_width: u32,
    /// 角度分类器输入目标高度。
    pub dst_height: u32,
}

impl Default for AngleConfig {
    fn default() -> Self {
        Self {
            dst_width: 192,
            dst_height: 32,
        }
    }
}

/// OCR 运行时总配置。
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// ncnn 使用的 CPU 线程数。
    pub num_threads: u32,
    /// 可选 Vulkan GPU 索引，`None` 表示禁用 GPU。
    pub gpu_index: Option<u32>,
    /// 文本检测相关配置。
    pub detector: DetectorConfig,
    /// 文本识别相关配置。
    pub recognizer: RecognizerConfig,
    /// 角度分类相关配置。
    pub angle: AngleConfig,
    /// 是否在结果中绘制文本框图像。
    pub draw_boxes: bool,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            gpu_index: None,
            detector: DetectorConfig::default(),
            recognizer: RecognizerConfig::default(),
            angle: AngleConfig::default(),
            draw_boxes: false,
        }
    }
}
