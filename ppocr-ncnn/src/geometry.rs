use crate::ocr::{Point, Quad};
use anyhow::Result;
use nalgebra::{SMatrix, SVector};
use std::cmp::Ordering;

/// 使用中心点、正交方向向量和半长半宽表示的旋转矩形。
#[derive(Debug, Clone, Copy)]
pub struct RotatedRect {
    pub center: Point,
    pub axis_u: Point,
    pub axis_v: Point,
    pub half_w: f32,
    pub half_h: f32,
}

impl RotatedRect {
    /// 返回矩形完整宽度。
    pub fn width(&self) -> f32 {
        self.half_w * 2.0
    }

    /// 返回矩形完整高度。
    pub fn height(&self) -> f32 {
        self.half_h * 2.0
    }

    /// 返回矩形周长。
    pub fn perimeter(&self) -> f32 {
        2.0 * (self.width() + self.height())
    }

    /// 返回矩形面积。
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// 将旋转矩形转换为标准顺序的四边形。
    pub fn to_quad(self) -> Quad {
        let ux = self.axis_u.x * self.half_w;
        let uy = self.axis_u.y * self.half_w;
        let vx = self.axis_v.x * self.half_h;
        let vy = self.axis_v.y * self.half_h;
        order_quad([
            Point::new(self.center.x - ux - vx, self.center.y - uy - vy),
            Point::new(self.center.x + ux - vx, self.center.y + uy - vy),
            Point::new(self.center.x + ux + vx, self.center.y + uy + vy),
            Point::new(self.center.x - ux + vx, self.center.y - uy + vy),
        ])
    }

    /// 在两个方向上各向外扩张 `distance`。
    pub fn expanded(&self, distance: f32) -> Self {
        Self {
            half_w: self.half_w + distance,
            half_h: self.half_h + distance,
            ..*self
        }
    }
}

/// 使用单调链算法计算点集凸包。
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| match a.x.total_cmp(&b.x) {
        Ordering::Equal => a.y.total_cmp(&b.y),
        other => other,
    });

    let mut lower = Vec::with_capacity(sorted.len());
    for point in &sorted {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::with_capacity(sorted.len());
    for point in sorted.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// 计算包围点集的最小面积旋转矩形。
pub fn min_area_rect(points: &[Point]) -> Option<RotatedRect> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        return Some(RotatedRect {
            center: points[0],
            axis_u: Point::new(1.0, 0.0),
            axis_v: Point::new(0.0, 1.0),
            half_w: 0.5,
            half_h: 0.5,
        });
    }

    let hull = convex_hull(points);
    if hull.len() == 2 {
        let center = midpoint(hull[0], hull[1]);
        let axis_u = normalize(sub(hull[1], hull[0]));
        let axis_v = Point::new(-axis_u.y, axis_u.x);
        let width = distance(hull[0], hull[1]).max(1.0);
        return Some(RotatedRect {
            center,
            axis_u,
            axis_v,
            half_w: width * 0.5,
            half_h: 0.5,
        });
    }

    let mut best: Option<(f32, RotatedRect)> = None;
    for i in 0..hull.len() {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % hull.len()];
        let edge = sub(p1, p0);
        let edge_len = (edge.x * edge.x + edge.y * edge.y).sqrt();
        if edge_len <= f32::EPSILON {
            continue;
        }
        let axis_u = Point::new(edge.x / edge_len, edge.y / edge_len);
        let axis_v = Point::new(-axis_u.y, axis_u.x);

        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        for point in &hull {
            let proj_u = dot(*point, axis_u);
            let proj_v = dot(*point, axis_v);
            min_u = min_u.min(proj_u);
            max_u = max_u.max(proj_u);
            min_v = min_v.min(proj_v);
            max_v = max_v.max(proj_v);
        }

        let width = (max_u - min_u).max(1.0);
        let height = (max_v - min_v).max(1.0);
        let area = width * height;
        let center = Point::new(
            axis_u.x * (min_u + max_u) * 0.5 + axis_v.x * (min_v + max_v) * 0.5,
            axis_u.y * (min_u + max_u) * 0.5 + axis_v.y * (min_v + max_v) * 0.5,
        );

        let rect = RotatedRect {
            center,
            axis_u,
            axis_v,
            half_w: width * 0.5,
            half_h: height * 0.5,
        };

        match best {
            Some((best_area, _)) if area >= best_area => {}
            _ => best = Some((area, rect)),
        }
    }

    best.map(|(_, rect)| rect)
}

/// 计算四边形区域内概率图的平均值。
pub fn polygon_mean(prob_map: &[f32], width: usize, height: usize, quad: &Quad) -> f32 {
    let n = quad.len(); // 对于 Quad 固定是 4

    // 单次遍历算 bounding box
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for p in quad.iter() {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let min_y = (min_y.floor() as i32).clamp(0, height as i32 - 1);
    let max_y = (max_y.ceil() as i32).clamp(0, height as i32 - 1);

    let mut sum = 0.0f32;
    let mut count = 0usize;

    for y in min_y..=max_y {
        let fy = y as f32 + 0.5;
        let mut xs = [0.0f32; 4];
        let mut hit = 0usize;

        for i in 0..n {
            let a = quad[i];
            let b = quad[(i + 1) % n];
            let (ay, by) = (a.y, b.y);
            if (ay <= fy && by > fy) || (by <= fy && ay > fy) {
                let t = (fy - ay) / (by - ay);
                xs[hit] = a.x + t * (b.x - a.x);
                hit += 1;
            }
        }

        if hit < 2 {
            continue;
        }

        xs[..hit].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let x0 = (xs[0].floor() as i32).clamp(0, width as i32 - 1);
        let x1 = (xs[hit - 1].ceil() as i32).clamp(0, width as i32 - 1);

        let row = &prob_map[y as usize * width..];
        for x in x0..=x1 {
            sum += row[x as usize];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// 将四边形顶点重排为左上、右上、右下、左下。
pub fn order_quad(quad: Quad) -> Quad {
    let mut points = quad;
    points.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

    let mut top = [points[0], points[1]];
    let mut bottom = [points[2], points[3]];
    top.sort_by(|a, b| a.x.total_cmp(&b.x));
    bottom.sort_by(|a, b| a.x.total_cmp(&b.x));
    [top[0], top[1], bottom[1], bottom[0]]
}

/// 将四边形限制在图像边界内，并规范顶点顺序。
pub fn clamp_quad(mut quad: Quad, width: u32, height: u32) -> Quad {
    let max_x = (width.saturating_sub(1)) as f32;
    let max_y = (height.saturating_sub(1)) as f32;
    for point in &mut quad {
        point.x = point.x.clamp(0.0, max_x);
        point.y = point.y.clamp(0.0, max_y);
    }
    order_quad(quad)
}

/// 使用上边和左边长度近似四边形的宽高。
pub fn quad_size(quad: &Quad) -> (u32, u32) {
    let width = distance(quad[0], quad[1]).round().max(1.0) as u32;
    let height = distance(quad[0], quad[3]).round().max(1.0) as u32;
    (width, height)
}

/// 返回四边形的几何中心。
pub fn quad_center(quad: &Quad) -> Point {
    let (sx, sy) = quad.iter().fold((0.0f32, 0.0f32), |(ax, ay), point| {
        (ax + point.x, ay + point.y)
    });
    Point::new(sx * 0.25, sy * 0.25)
}

/// 计算两点之间的欧氏距离。
pub fn distance(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// 求解将矩形映射到目标四边形的单应矩阵。
pub fn homography_from_quad_to_rect(quad: &Quad, width: f32, height: f32) -> Result<[[f32; 3]; 3]> {
    let src = [
        (0.0f32, 0.0f32),
        (width, 0.0),
        (width, height),
        (0.0, height),
    ];
    let dst = [
        (quad[0].x, quad[0].y),
        (quad[1].x, quad[1].y),
        (quad[2].x, quad[2].y),
        (quad[3].x, quad[3].y),
    ];

    let mut a = SMatrix::<f32, 8, 8>::zeros();
    let mut b = SVector::<f32, 8>::zeros();
    for (row, ((x, y), (u, v))) in src.iter().zip(dst.iter()).enumerate() {
        let r0 = row * 2;
        let r1 = r0 + 1;

        a[(r0, 0)] = *x;
        a[(r0, 1)] = *y;
        a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -x * u;
        a[(r0, 7)] = -y * u;
        b[r0] = *u;

        a[(r1, 3)] = *x;
        a[(r1, 4)] = *y;
        a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -x * v;
        a[(r1, 7)] = -y * v;
        b[r1] = *v;
    }

    let h = a
        .lu()
        .solve(&b)
        .ok_or_else(|| anyhow::anyhow!("failed to solve perspective transform"))?;

    Ok([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

/// 使用单应矩阵变换二维点。
#[allow(dead_code)]
pub fn transform_point(h: &[[f32; 3]; 3], x: f32, y: f32) -> Point {
    let denom = h[2][0] * x + h[2][1] * y + h[2][2];
    let inv = if denom.abs() <= f32::EPSILON {
        0.0
    } else {
        1.0 / denom
    };
    Point::new(
        (h[0][0] * x + h[0][1] * y + h[0][2]) * inv,
        (h[1][0] * x + h[1][1] * y + h[1][2]) * inv,
    )
}

fn dot(a: Point, b: Point) -> f32 {
    a.x * b.x + a.y * b.y
}

fn cross(a: Point, b: Point, c: Point) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn sub(a: Point, b: Point) -> Point {
    Point::new(a.x - b.x, a.y - b.y)
}

fn normalize(point: Point) -> Point {
    let len = (point.x * point.x + point.y * point.y).sqrt();
    if len <= f32::EPSILON {
        Point::new(1.0, 0.0)
    } else {
        Point::new(point.x / len, point.y / len)
    }
}

fn midpoint(a: Point, b: Point) -> Point {
    Point::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)
}
