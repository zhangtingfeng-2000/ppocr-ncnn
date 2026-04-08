use crate::geometry::{RotatedRect, clamp_quad, min_area_rect, polygon_mean, quad_center};
use crate::ocr::config::DetectorConfig;
use crate::ocr::types::{Point, ScaleParam, TextBox};
use image::RgbImage;

/// 根据目标最长边计算检测阶段的缩放参数。
pub(super) fn compute_scale_param(src: &RgbImage, target_size: u32) -> ScaleParam {
    let src_width = src.width();
    let src_height = src.height();
    let ratio = if src_width > src_height {
        target_size as f32 / src_width as f32
    } else {
        target_size as f32 / src_height as f32
    };

    let mut dst_width = ((src_width as f32) * ratio) as u32;
    let mut dst_height = ((src_height as f32) * ratio) as u32;
    dst_width = dst_width.max(32) / 32 * 32;
    dst_height = dst_height.max(32) / 32 * 32;
    if dst_width == 0 {
        dst_width = 32;
    }
    if dst_height == 0 {
        dst_height = 32;
    }

    ScaleParam {
        src_width,
        src_height,
        dst_width,
        dst_height,
        ratio_width: dst_width as f32 / src_width.max(1) as f32,
        ratio_height: dst_height as f32 / src_height.max(1) as f32,
    }
}

/// 将检测模型输出的概率图转换为文本框结果。
pub(super) fn postprocess_boxes(
    prob_map: &[f32],
    width: usize,
    height: usize,
    scale: &ScaleParam,
    cfg: &DetectorConfig,
) -> anyhow::Result<Vec<TextBox>> {
    let mut binary = prob_map
        .iter()
        .map(|&value| u8::from(value > cfg.box_thresh))
        .collect::<Vec<_>>();
    let components = connected_components(&mut binary, width, height);
    let mut boxes = Vec::with_capacity(components.len());

    for points in components {
        if points.len() < 3 {
            continue;
        }
        let Some(rect) = min_area_rect(&points) else {
            continue;
        };
        if rect.width().min(rect.height()) < 3.0 {
            continue;
        }

        let base_quad = rect.to_quad();
        let score = polygon_mean(prob_map, width, height, &base_quad);
        if score < cfg.box_score_thresh {
            continue;
        }

        let expanded = expand_rect(rect, cfg.unclip_ratio);
        if expanded.width().min(expanded.height()) < 5.0 {
            continue;
        }

        let mut quad = expanded.to_quad();
        for point in &mut quad {
            point.x = (point.x / scale.ratio_width).clamp(0.0, scale.src_width as f32);
            point.y = (point.y / scale.ratio_height).clamp(0.0, scale.src_height as f32);
        }
        let quad = clamp_quad(quad, scale.src_width, scale.src_height);
        boxes.push(TextBox { quad, score });
    }

    boxes.sort_by(|a, b| {
        let ca = quad_center(&a.quad);
        let cb = quad_center(&b.quad);
        if (ca.y - cb.y).abs() > 10.0 {
            ca.y.total_cmp(&cb.y)
        } else {
            ca.x.total_cmp(&cb.x)
        }
    });
    Ok(boxes)
}

/// 在二值图上查找 8 邻域连通区域。
fn connected_components(binary: &mut [u8], width: usize, height: usize) -> Vec<Vec<Point>> {
    let mut components = Vec::new();
    let mut stack = Vec::new();

    for idx in 0..binary.len() {
        if binary[idx] != 1 {
            continue;
        }

        stack.clear();
        binary[idx] = 2;
        stack.push((idx % width, idx / width));

        let mut component = Vec::new();
        while let Some((seed_x, y)) = stack.pop() {
            let row_start = y * width;
            if binary[row_start + seed_x] == 0 {
                continue;
            }

            let mut x0 = seed_x;
            while x0 > 0 && binary[row_start + x0 - 1] != 0 {
                x0 -= 1;
            }

            let mut x1 = seed_x;
            while x1 + 1 < width && binary[row_start + x1 + 1] != 0 {
                x1 += 1;
            }

            for x in x0..=x1 {
                binary[row_start + x] = 2;
                if is_boundary_pixel(binary, width, height, x, y, x0, x1) {
                    component.push(Point::new(x as f32, y as f32));
                }
            }

            let scan_start = x0.saturating_sub(1);
            let scan_end = (x1 + 1).min(width - 1);
            for ny in [y.checked_sub(1), (y + 1 < height).then_some(y + 1)]
                .into_iter()
                .flatten()
            {
                let neighbor_row = ny * width;
                let mut x = scan_start;
                while x <= scan_end {
                    let nidx = neighbor_row + x;
                    if binary[nidx] != 1 {
                        x += 1;
                        continue;
                    }
                    binary[nidx] = 2;
                    stack.push((x, ny));
                    x += 1;
                    while x <= scan_end && binary[neighbor_row + x] != 0 {
                        x += 1;
                    }
                }
            }
        }

        components.push(component);
    }

    components
}

fn is_boundary_pixel(
    binary: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    x0: usize,
    x1: usize,
) -> bool {
    x == x0
        || x == x1
        || y == 0
        || y + 1 == height
        || binary[(y - 1) * width + x] == 0
        || binary[(y + 1) * width + x] == 0
}

/// 按 DB 类检测思路对旋转矩形进行外扩。
fn expand_rect(rect: RotatedRect, ratio: f32) -> RotatedRect {
    let perimeter = rect.perimeter().max(1.0);
    let distance = rect.area() * ratio / perimeter;
    rect.expanded(distance)
}
