use image::RgbImage;

/// 图像坐标系中的二维点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    /// 使用 `x` 和 `y` 坐标创建一个点。
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// 一个按左上、右上、右下、左下顺序排列的四边形。
pub type Quad = [Point; 4];

/// 检测缩放时的源图与目标图几何参数。
#[derive(Debug, Clone)]
pub struct ScaleParam {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub ratio_width: f32,
    pub ratio_height: f32,
}

/// 识别前的候选文本框。
#[derive(Debug, Clone)]
pub struct TextBox {
    pub quad: Quad,
    pub score: f32,
}

/// 文本块角度分类结果。
#[derive(Debug, Clone)]
pub struct Angle {
    pub index: usize,
    pub score: f32,
}

/// 单个文本块识别结果。
#[derive(Debug, Clone)]
pub struct TextLine {
    pub text: String,
    pub char_scores: Vec<f32>,
}

/// 单个检测文本块的最终 OCR 结果。
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub quad: Quad,
    pub box_score: f32,
    pub angle: Option<Angle>,
    pub text: String,
    pub char_scores: Vec<f32>,
}

/// 单次 OCR 的耗时拆分，单位为毫秒。
#[derive(Debug, Clone, Copy, Default)]
pub struct OcrTiming {
    pub total_ms: f64,
    pub prepare_ms: f64,
    pub detect_ms: f64,
    pub crop_ms: f64,
    pub angle_cls_ms: f64,
    pub recognize_ms: f64,
    pub postprocess_ms: f64,
}

/// 单张图片的 OCR 汇总结果。
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// 按阅读顺序排列的文本块结果。
    pub blocks: Vec<TextBlock>,
    /// 所有识别结果按换行拼接后的文本。
    pub text: String,
    /// 开启绘框时返回的标注图。
    pub box_image: Option<RgbImage>,
    /// OCR 流程耗时信息。
    pub timing: OcrTiming,
}
