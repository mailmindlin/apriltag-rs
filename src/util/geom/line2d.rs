use crate::util::math::{Vec2, FMA, Vec2Builder};

use super::Point2D;

pub(crate) struct Line2D {
    p: Point2D,
    u: Point2D, // Always a unit vector
}

impl Line2D {
    pub fn from_points(p0: Point2D, p1: Point2D) -> Self {
        let p = p0;
        let u = Point2D::from_vec((p1 - &p0).norm());
        Self { p, u }
    }

    pub fn get_coordinate(&self, q: &Point2D) -> f64 {
        (q - &self.p).dot(*self.u.vec())
    }

    // Compute intersection of two line segments. If they intersect,
    // result is stored in p and 1 is returned. Otherwise, zero is
    // returned. p may be NULL.
    pub fn intersect_line(&self, other: &Line2D) -> Option<Point2D> {
        // this implementation is many times faster than the original,
        // mostly due to avoiding a general-purpose LU decomposition in
        // Matrix.inverse().
        let m00 = self.u.x();
        let m01= -other.u.x();
        let m10 = self.u.y();
        let m11= -other.u.y();

        // determinant of m
        let det = m00*m11-m01*m10;

        // parallel lines?
        if det.abs() < 1e-8 {
            return None;
        }

        // inverse of m
        let i = Vec2::of(m11, -m01) / det;

        let b = other.p - &self.p;

        let x00 = i.dot(b);

        Some(Point2D::from_vec(
            self.p.vec().fma(
                *self.u.vec(),
                Vec2::dup(x00)
            )
        ))
    }
}

pub(crate) struct LineSegment2D {
    line: Line2D,
    p1: Point2D,
}

impl LineSegment2D {
    pub fn from_points(p0: Point2D, p1: Point2D) -> Self {
        Self {
            line: Line2D::from_points(p0, p1),
            p1,
        }
    }

    /// Find the point p on segment seg that is closest to point q.
    pub fn closest_point(&self, q: &Point2D) -> Point2D {
        let a = self.line.get_coordinate(&self.line.p);
        let b = self.line.get_coordinate(&self.p1);
        let c = self.line.get_coordinate(q);

        let c = if a < b {
            c.clamp(a, b)
        } else {
            c.clamp(b, a)
        };

        Point2D::from_vec(
            self.line.p.vec()
            .fma(
                *self.line.u.vec(),
                Vec2::dup(c)
            )
        )
    }

    // Compute intersection of two line segments. If they intersect,
    // result is stored in p and 1 is returned. Otherwise, zero is
    // returned. p may be NULL.
    pub fn intersect_segment(&self, other: &LineSegment2D) -> Option<Point2D> {
        let tmp = if let Some(tmp) = self.intersect_line(&other.line) {
            tmp
        } else {
            return None;
        };

        let a = other.line.get_coordinate(&other.line.p);
        let b = other.line.get_coordinate(&other.p1);
        let c = other.line.get_coordinate(&tmp);

        // does intersection lie on second line?
        if (c<a && c<b) || (c>a && c>b) {
            return None;
        }

        Some(tmp)
    }

    // Compute intersection of a line segment and a line. If they
    // intersect, result is stored in p and 1 is returned. Otherwise, zero
    // is returned. p may be NULL.
    pub fn intersect_line(&self, line: &Line2D) -> Option<Point2D> {
        let tmp = if let Some(tmp) = self.line.intersect_line(line) {
            tmp
        } else {
            return None;
        };

        let a = self.line.get_coordinate(&self.line.p);
        let b = self.line.get_coordinate(&self.p1);
        let c = self.line.get_coordinate(&tmp);

        // does intersection lie on the first line?
        if (c<a && c<b) || (c>a && c>b) {
            return None;
        }

        Some(tmp)
    }
}