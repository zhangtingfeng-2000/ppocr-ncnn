use crate::geometry::{homography_from_quad_to_rect, quad_size};
use crate::ocr::{Point, Quad};
use image::{Rgb, RgbImage, imageops};
use rayon::prelude::*;

/// 为输入图像四周补白边。
pub fn make_padding(src: &RgbImage, padding: u32) -> RgbImage {
    if padding == 0 {
        return src.clone();
    }
    let mut out = RgbImage::from_pixel(
        src.width() + padding * 2,
        src.height() + padding * 2,
        Rgb([255, 255, 255]),
    );
    imageops::overlay(&mut out, src, padding as i64, padding as i64);
    out
}

/// 从原图中裁剪四边形区域，并进行透视矫正。
///
/// 当裁剪结果过高过窄时，会逆时针旋转，便于后续识别模型按横排文本处理。
pub fn crop_quad(src: &RgbImage, quad: &Quad) -> anyhow::Result<RgbImage> {
    let (width, height) = quad_size(quad);
    let width = width.max(1);
    let height = height.max(1);
    let h = homography_from_quad_to_rect(quad, width as f32, height as f32)?;

    let src_width = src.width() as usize;
    let src_height = src.height() as usize;
    let src_last_x = src_width.saturating_sub(1);
    let src_last_y = src_height.saturating_sub(1);
    let max_x = src_last_x as f32;
    let max_y = src_last_y as f32;
    let src_stride = src_width * 3;
    let src_raw = src.as_raw();

    let out_width = width as usize;
    let out_height = height as usize;
    let out_stride = out_width * 3;
    let mut out_raw = vec![0u8; out_stride * out_height];

    let h00 = h[0][0];
    let h01 = h[0][1];
    let h02 = h[0][2];
    let h10 = h[1][0];
    let h11 = h[1][1];
    let h12 = h[1][2];
    let h20 = h[2][0];
    let h21 = h[2][1];
    let h22 = h[2][2];

    // 按行分块并行，每行独立计算
    out_raw
        .chunks_exact_mut(out_stride)
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(y, row)| {
            let fy = y as f32;
            // 每行起始的增量值
            let mut u_num = h01 * fy + h02;
            let mut v_num = h11 * fy + h12;
            let mut denom = h21 * fy + h22;

            for x in 0..out_width {
                let inv = if denom.abs() <= f32::EPSILON {
                    0.0
                } else {
                    1.0 / denom
                };
                let src_x = (u_num * inv).clamp(0.0, max_x);
                let src_y = (v_num * inv).clamp(0.0, max_y);

                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(src_last_x);
                let y1 = (y0 + 1).min(src_last_y);

                // 定点数插值权重，Q8 精度（0..=256）
                let wx = ((src_x - x0 as f32) * 256.0) as u32;
                let wy = ((src_y - y0 as f32) * 256.0) as u32;
                let wx_inv = 256 - wx;
                let wy_inv = 256 - wy;

                let p00 = y0 * src_stride + x0 * 3;
                let p10 = y0 * src_stride + x1 * 3;
                let p01 = y1 * src_stride + x0 * 3;
                let p11 = y1 * src_stride + x1 * 3;
                let out_idx = x * 3;

                unsafe {
                    *row.get_unchecked_mut(out_idx) = interp_channel(
                        src_raw, p00, p10, p01, p11, wx, wx_inv, wy, wy_inv, 0,
                    );
                    *row.get_unchecked_mut(out_idx + 1) = interp_channel(
                        src_raw, p00, p10, p01, p11, wx, wx_inv, wy, wy_inv, 1,
                    );
                    *row.get_unchecked_mut(out_idx + 2) = interp_channel(
                        src_raw, p00, p10, p01, p11, wx, wx_inv, wy, wy_inv, 2,
                    );
                }

                u_num += h00;
                v_num += h10;
                denom += h20;
            }
        });

    let out = RgbImage::from_raw(width, height, out_raw)
        .ok_or_else(|| anyhow::anyhow!("failed to construct cropped image"))?;

    if out.height() as f32 >= out.width() as f32 * 1.5 {
        Ok(rotate_ccw(&out))
    } else {
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn interp_channel(
    src_raw: &[u8],
    p00: usize,
    p10: usize,
    p01: usize,
    p11: usize,
    wx: u32,
    wx_inv: u32,
    wy: u32,
    wy_inv: u32,
    channel: usize,
) -> u8 {
    unsafe {
        let top = *src_raw.get_unchecked(p00 + channel) as u32 * wx_inv
            + *src_raw.get_unchecked(p10 + channel) as u32 * wx;
        let bottom = *src_raw.get_unchecked(p01 + channel) as u32 * wx_inv
            + *src_raw.get_unchecked(p11 + channel) as u32 * wx;
        ((top * wy_inv + bottom * wy) >> 16) as u8
    }
}

/// 将图像缩放到分类器目标高度，再按目标宽度进行补白或裁剪。
pub fn adjust_target_img(src: &RgbImage, dst_width: u32, dst_height: u32) -> RgbImage {
    let scale = dst_height as f32 / src.height().max(1) as f32;
    let scaled_width = ((src.width() as f32 * scale).round() as u32).max(1);
    let resized = imageops::resize(
        src,
        scaled_width,
        dst_height,
        imageops::FilterType::Triangle,
    );
    let mut out = RgbImage::from_pixel(dst_width, dst_height, Rgb([255, 255, 255]));
    if resized.width() <= dst_width {
        imageops::overlay(&mut out, &resized, 0, 0);
    } else {
        let cropped = imageops::crop_imm(&resized, 0, 0, dst_width, dst_height).to_image();
        imageops::overlay(&mut out, &cropped, 0, 0);
    }
    out
}

/// 将图像旋转 180 度。
pub fn rotate_180(src: &RgbImage) -> RgbImage {
    imageops::rotate180(src)
}

/// 在图像上绘制四边形边框。
pub fn draw_quad(img: &mut RgbImage, quad: &Quad, color: Rgb<u8>) {
    for i in 0..4 {
        let start = quad[i];
        let end = quad[(i + 1) % 4];
        draw_line(img, start, end, color);
    }
}

fn draw_line(img: &mut RgbImage, start: Point, end: Point, color: Rgb<u8>) {
    let mut x0 = start.x.round() as i32;
    let mut y0 = start.y.round() as i32;
    let x1 = end.x.round() as i32;
    let y1 = end.y.round() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        if x0 >= 0 && y0 >= 0 && x0 < img.width() as i32 && y0 < img.height() as i32 {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err * 2;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn rotate_ccw(src: &RgbImage) -> RgbImage {
    let src_width = src.width() as usize;
    let src_height = src.height() as usize;
    let src_stride = src_width * 3;
    let out_width = src_height;
    let out_height = src_width;
    let out_stride = out_width * 3;

    let src_raw = src.as_raw();
    let mut out_raw = vec![0u8; out_stride * out_height];

    for y in 0..src_height {
        for x in 0..src_width {
            let src_idx = y * src_stride + x * 3;
            let dst_x = y;
            let dst_y = src_width - 1 - x;
            let dst_idx = dst_y * out_stride + dst_x * 3;
            out_raw[dst_idx..dst_idx + 3].copy_from_slice(&src_raw[src_idx..src_idx + 3]);
        }
    }

    RgbImage::from_raw(src.height(), src.width(), out_raw)
        .expect("rotate_ccw output dimensions must match buffer length")
}
