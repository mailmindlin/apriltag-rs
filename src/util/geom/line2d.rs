use crate::util::math::{Vec2, FMA, Vec2Builder};

use super::Point2D;

pub(crate) struct Line2D {
    p: Point2D,
    u: Point2D, // Always a unit vector
}

impl Line2D {
    pub fn from_points(p0: Point2D, p1: Point2D) -> Self {
        let p = p0;
        let u = Point2D::from_vec((p1 - p0).norm());
        Self { p, u }
    }

    pub fn get_coordinate(&self, q: Point2D) -> f64 {
        (q - self.p).dot(self.u.vec())
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

        let b = other.p - self.p;

        let x00 = i.dot(b);

        Some(Point2D::from_vec(
            self.p.vec().fma(
                self.u.vec(),
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
    pub fn closest_point(&self, q: Point2D) -> Point2D {
        let a = self.line.get_coordinate(self.line.p);
        let b = self.line.get_coordinate(self.p1);
        let c = self.line.get_coordinate(q);

        let c = if a < b {
            c.clamp(a, b)
        } else {
            c.clamp(b, a)
        };

        Point2D::from_vec(
            self.line.p.vec()
            .fma(
                self.line.u.vec(),
                Vec2::dup(c)
            )
        )
    }

    // Compute intersection of two line segments. If they intersect,
    // result is stored in p and 1 is returned. Otherwise, zero is
    // returned. p may be NULL.
    pub fn intersect_segment(&self, other: &LineSegment2D) -> Option<Point2D> {
        let tmp = self.intersect_line(&other.line)?;

        let a = other.line.get_coordinate(other.line.p);
        let b = other.line.get_coordinate(other.p1);
        let c = other.line.get_coordinate(tmp);

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
        let tmp = self.line.intersect_line(line)?;

        let a = self.line.get_coordinate(self.line.p);
        let b = self.line.get_coordinate(self.p1);
        let c = self.line.get_coordinate(tmp);

        // does intersection lie on the first line?
        if (c<a && c<b) || (c>a && c>b) {
            return None;
        }

        Some(tmp)
    }
}

#[cfg(test)]
mod test {
    use super::{Line2D, LineSegment2D};
    use crate::util::geom::Point2D;

    const EPS: f64 = 1e-10;

    fn pt(x: f64, y: f64) -> Point2D { Point2D::of(x, y) }

    fn assert_pt_close(a: Point2D, b: Point2D) {
        assert!((a.x() - b.x()).abs() < EPS && (a.y() - b.y()).abs() < EPS,
            "expected {:?}, got {:?}", b, a);
    }

    // --- Line2D::intersect_line ---

    #[test]
    fn intersect_line_basic() {
        // horizontal y=0 and vertical x=0.5 → intersect at (0.5, 0)
        let h = Line2D::from_points(pt(0., 0.), pt(1., 0.));
        let v = Line2D::from_points(pt(0.5, -1.), pt(0.5, 1.));
        let p = h.intersect_line(&v).expect("lines should intersect");
        assert_pt_close(p, pt(0.5, 0.));
    }

    #[test]
    fn intersect_line_diagonal() {
        // y=x and y=2-x → intersect at (1, 1)
        let l1 = Line2D::from_points(pt(0., 0.), pt(2., 2.));
        let l2 = Line2D::from_points(pt(0., 2.), pt(2., 0.));
        let p = l1.intersect_line(&l2).expect("lines should intersect");
        assert_pt_close(p, pt(1., 1.));
    }

    #[test]
    fn intersect_line_parallel_returns_none() {
        let l1 = Line2D::from_points(pt(0., 0.), pt(1., 0.));
        let l2 = Line2D::from_points(pt(0., 1.), pt(1., 1.));
        assert!(l1.intersect_line(&l2).is_none());
    }

    #[test]
    fn intersect_line_symmetric() {
        // intersection of A with B == intersection of B with A
        let l1 = Line2D::from_points(pt(0., 0.), pt(3., 1.));
        let l2 = Line2D::from_points(pt(1., 0.), pt(0., 3.));
        let p1 = l1.intersect_line(&l2).unwrap();
        let p2 = l2.intersect_line(&l1).unwrap();
        assert_pt_close(p1, p2);
    }

    // --- LineSegment2D::closest_point ---

    #[test]
    fn closest_point_projects_to_interior() {
        let seg = LineSegment2D::from_points(pt(0., 0.), pt(2., 0.));
        let q = pt(1., 5.);
        assert_pt_close(seg.closest_point(q), pt(1., 0.));
    }

    #[test]
    fn closest_point_clamped_to_start() {
        let seg = LineSegment2D::from_points(pt(0., 0.), pt(2., 0.));
        assert_pt_close(seg.closest_point(pt(-3., 1.)), pt(0., 0.));
    }

    #[test]
    fn closest_point_clamped_to_end() {
        let seg = LineSegment2D::from_points(pt(0., 0.), pt(2., 0.));
        assert_pt_close(seg.closest_point(pt(5., 1.)), pt(2., 0.));
    }

    #[test]
    fn closest_point_reversed_segment() {
        // Segment stored in reverse order should still clamp correctly
        let seg = LineSegment2D::from_points(pt(2., 0.), pt(0., 0.));
        assert_pt_close(seg.closest_point(pt(1., 7.)), pt(1., 0.));
        assert_pt_close(seg.closest_point(pt(-1., 7.)), pt(0., 0.));
        assert_pt_close(seg.closest_point(pt(5., 7.)), pt(2., 0.));
    }

    // --- LineSegment2D::intersect_segment ---

    #[test]
    fn intersect_segment_crossing() {
        // X-crossing: (0,0)-(1,1) and (0,1)-(1,0)
        let s1 = LineSegment2D::from_points(pt(0., 0.), pt(1., 1.));
        let s2 = LineSegment2D::from_points(pt(0., 1.), pt(1., 0.));
        let p = s1.intersect_segment(&s2).expect("segments cross");
        assert_pt_close(p, pt(0.5, 0.5));
    }

    #[test]
    fn intersect_segment_t_shape() {
        // T: (0,1)-(2,1) crossed by (1,0)-(1,1) at exactly (1,1)
        let horiz = LineSegment2D::from_points(pt(0., 1.), pt(2., 1.));
        let vert  = LineSegment2D::from_points(pt(1., 0.), pt(1., 1.));
        assert!(horiz.intersect_segment(&vert).is_some());
    }

    #[test]
    fn intersect_segment_miss_parallel() {
        let s1 = LineSegment2D::from_points(pt(0., 0.), pt(1., 0.));
        let s2 = LineSegment2D::from_points(pt(0., 1.), pt(1., 1.));
        assert!(s1.intersect_segment(&s2).is_none());
    }

    #[test]
    fn intersect_segment_miss_collinear_gap() {
        // Same line, non-overlapping
        let s1 = LineSegment2D::from_points(pt(0., 0.), pt(1., 0.));
        let s2 = LineSegment2D::from_points(pt(2., 0.), pt(3., 0.));
        assert!(s1.intersect_segment(&s2).is_none());
    }

    #[test]
    fn intersect_segment_lines_cross_but_segments_miss() {
        // Lines y=x and y=2-x cross at (1,1), but segments don't reach it
        let s1 = LineSegment2D::from_points(pt(0., 0.), pt(0.5, 0.5));
        let s2 = LineSegment2D::from_points(pt(0., 2.), pt(0.5, 1.5));
        assert!(s1.intersect_segment(&s2).is_none());
    }
}