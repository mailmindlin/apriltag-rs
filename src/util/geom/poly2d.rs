use std::ops::{Index, IndexMut};

use crate::util::math::{Vec2, mod2pi};

use super::{Point2D, LineSegment2D, Line2D};

pub(crate) struct Poly2D(pub Vec<Point2D>);

impl Index<usize> for Poly2D {
    type Output = Point2D;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Poly2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl IntoIterator for Poly2D {
    type Item = Point2D;
    type IntoIter = std::vec::IntoIter<Point2D>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Poly2D {
    pub fn new() -> Self {
        Poly2D(Vec::new())
    }

    pub fn of(pts: &[Point2D]) -> Self {
        Self(pts.to_vec())
    }
    
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn zeroes(size: usize) -> Self {
        Self(vec![Point2D::zero(); size])
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }
    pub fn add(&mut self, point: Point2D) {
        self.0.push(point);
    }
    pub fn make_ccw(&mut self) {
        // Step one: we want the points in counter-clockwise order.
        // If the points are in clockwise order, we'll reverse them.
        let mut total_theta = 0.;
        let mut last_theta = 0.;

        // Count the angle accumulated going around the polygon. If
        // the sum is +2pi, it's CCW. Otherwise, we'll get -2pi.
        let sz = self.len();

        for i in 0..=sz {
            let p0 = &self[i % sz];
            let p1 = &self[(i + 1) % sz];

            let this_theta = p0.angle_to(p1);

            if i > 0 {
                let dtheta = mod2pi(this_theta-last_theta);
                total_theta += dtheta;
            }

            last_theta = this_theta;
        }

        let ccw = total_theta > 0.;

        // reverse order if necessary.
        if !ccw {
            for i in 0..(sz/2) {
                self.swap(i, sz - 1 - i);
            }
        }
    }

    /*pub fn contains_point(&self, q: &Point2D) -> bool { // g2d_polygon_contains_point_ref
        // use winding. If the point is inside the polygon, we'll wrap
        // around it (accumulating 6.28 radians). If we're outside the
        // polygon, we'll accumulate zero.
        let psz = self.len();

        let mut acc_theta = 0;

        let mut last_theta;

        for i in 0..=psz {
            let p = &self[i % psz];

            let this_theta = p.angle_to(q);

            if i != 0 {
                acc_theta += mod2pi(this_theta - last_theta);
            }

            last_theta = this_theta;
        }

        return acc_theta > std::f64::consts::PI;
    }*/

    /// creates and returns a Poly2D. The resulting polygon is
    /// CCW and implicitly closed. Unnecessary colinear points are omitted.
    pub fn convex_hull(&self) -> Poly2D {
        // gift-wrap algorithm.

        // step 1: find left most point.
        let insz = self.len();

        // must have at least 2 points. (XXX need 3?)
        assert!(insz >= 2);

        let pleft = self.0.iter()
            .min_by(|p, q| f64::total_cmp(&p.x(), &q.x()))
            .unwrap(); // cannot be None since there must be at least one point.

        let mut hull = Poly2D::new();
        hull.add(*pleft);

        // step 2. gift wrap. Keep searching for points that make the
        // smallest-angle left-hand turn. This implementation is carefully
        // written to use only addition/subtraction/multiply. No division
        // or sqrts. This guarantees exact results for integer-coordinate
        // polygons (no rounding/precision problems).
        let mut p = pleft;
        loop {
            let mut q = None;
            // the normal to the line (p, q) (not necessarily unit length).
            let mut n = Vec2::zero();

            // Search for the point q for which the line (p,q) is most "to
            // the right of" the other points. (i.e., every time we find a
            // point that is to the right of our current line, we change
            // lines.)
            for thisq in self.0.iter() {
                if thisq == p {
                    continue;
                }

                // the first time we find another point, we initialize our
                // value of q, forming the line (p,q)
                if q.is_none() {
                    q = Some(thisq);
                    n = Vec2::of(
                        thisq.y() - p.y(),
                        p.x() - thisq.x(),
                    );
                } else {
                    // we already have a line (p,q). is point thisq RIGHT OF line (p, q)?
                    let e = thisq - &p;
                    let dot = e.dot(&n);

                    if dot > 0. {
                        // it is. change our line.
                        q = Some(thisq);
                        n = Vec2::of(
                            thisq.y() - p.y(),
                            p.x() - thisq.x()
                        );
                    }
                }
            }

            // we must have elected *some* line, so long as there are at
            // least 2 points in the polygon.
            let q = q.unwrap();

            // loop completed?
            if q == pleft {
                break;
            }

            let mut colinear = false;

            // is this new point colinear with the last two?
            if hull.len() > 1 {
                let o = &hull[hull.len() - 2];

                let e = o - &p;

                if n.dot(&e) == 0. {
                    colinear = true;
                }
            }

            // if it is colinear, overwrite the last one.
            if colinear {
                let len = hull.len();
                hull.0[len - 1] = *q;
            } else {
                hull.add(*q);
            }

            p = q;
        }

        return hull;
    }

    // Find point p on the boundary of poly that is closest to q.
    pub fn closest_boundary_point(&self, q: &Point2D) -> Option<Point2D> {
        let mut min_dist = f64::INFINITY;
        let mut result = None;

        for i in 0..self.len() {
            let p0 = &self[i];
            let p1 = &self[(i + 1) % self.len()];

            let seg = LineSegment2D::from_points(*p0, *p1);

            let thisp = seg.closest_point(q);

            let dist = q.distance_to(&thisp);
            if dist < min_dist {
                result = Some(thisp);
                min_dist = dist;
            }
        }

        result
    }

    pub fn contains_point(&self, q: &Point2D) -> bool {
        // use winding. If the point is inside the polygon, we'll wrap
        // around it (accumulating 6.28 radians). If we're outside the
        // polygon, we'll accumulate zero.
        let psz = self.len();
        assert!(psz > 0);

        let mut last_quadrant = None;
        let mut quad_acc = 0;

        for i in 0..=psz {
            let p = &self[i % psz];

            // p[0] < q[0]       p[1] < q[1]    quadrant
            //     0                 0              0
            //     0                 1              3
            //     1                 0              1
            //     1                 1              2

            // p[1] < q[1]       p[0] < q[0]    quadrant
            //     0                 0              0
            //     0                 1              1
            //     1                 0              3
            //     1                 1              2

            let quadrant =if p.x() < q.x() {
                if p.y() < q.y() {
                    2
                } else {
                    1
                }
            } else {
                if p.y() < q.y() {
                    3
                } else {
                    0
                }
            };

            if let Some(last_quadrant) = last_quadrant { // i > 0
                let dquadrant = quadrant - last_quadrant;

                // encourage a jump table by mapping to small positive integers.
                match dquadrant {
                    -3 | 1 => {
                        quad_acc += 1;
                    },
                    -1 | 3 => {
                        quad_acc -= 1;
                    },
                    0 => {},
                    -2 | 2 => {
                        // get the previous point.
                        let p0 = &self[i-1];

                        // Consider the points p0 and p (the points around the
                        //polygon that we are tracing) and the query point q.
                        //
                        // If we've moved diagonally across quadrants, we want
                        // to measure whether we have rotated +PI radians or
                        // -PI radians. We can test this by computing the dot
                        // product of vector (p0-q) with the vector
                        // perpendicular to vector (p-q)
                        let n = Vec2::of(
                            p.y() - q.y(),
                            q.x() - p.x(),
                        );

                        let dot = n.dot(&(p0 - &q));
                        if dot < 0. {
                            quad_acc -= 2;
                        } else {
                            quad_acc += 2;
                        }
                    },
                    _ => unreachable!(),
                }
            }

            last_quadrant = Some(quadrant);
        }

        let v = (quad_acc >= 2) || (quad_acc <= -2);

        if false && v != self.contains_point(q) {
            panic!("Failure {} {}", v, quad_acc);
        }

        v
    }

    // do the edges of polya and polyb collide? (Does NOT test for containment).
    pub fn intersects_polygon(&self, other: &Poly2D) -> bool {
        // do any of the line segments collide? If so, the answer is no.

        // dumb N^2 method.
        for ia in 0..self.len() {
            let pa0 = &self[ia];
            let pa1 = &self[(ia + 1) % self.len()];

            let sega = LineSegment2D::from_points(*pa0, *pa1);
            for ib in 0..other.len() {
                let pb0 = &other[ib];
                let pb1 = &other[(ib + 1) % other.len()];

                let segb = LineSegment2D::from_points(*pb0, *pb1);

                if sega.intersect_segment(&segb).is_some() {
                    return true;
                }
            }
        }

        false
    }

    // does polya completely contain polyb?
    fn contains_polygon(&self, other: &Poly2D) -> bool {
        // do any of the line segments collide? If so, the answer is no.
        if self.intersects_polygon(other) {
            return false;
        }

        // if none of the edges cross, then the polygon is either fully
        // contained or fully outside.
        self.contains_point(&other[0])
    }

    /// compute a point that is inside the polygon. (It may not be *far* inside though)
    pub fn get_interior_point(&self) -> Point2D {
        // take the first three points, which form a triangle. Find the middle point
        let a = self[0];
        let b = self[1];
        let c = self[2];

        Point2D::from_vec((a.vec() + b.vec() + c.vec()) / 3.)
    }

    pub fn overlaps_polygon(&self, other: &Poly2D) -> bool {
        // do any of the line segments collide? If so, the answer is yes.
        if self.intersects_polygon(other) {
            return true;
        }

        // if none of the edges cross, then the polygon is either fully
        // contained or fully outside.
        let p = other.get_interior_point();
        if self.contains_point(&p) {
            return true;
        }

        let p = self.get_interior_point();

        if other.contains_point(&p) {
            return true;
        }

        return false;
    }
    /// Returns the x intercepts
    pub fn rasterize(&self, y: f64) -> Vec<f64> {
        let sz = self.len();

        let line = {
            let p0 = Point2D::of(0., y);
            let p1 = Point2D::of(1., y);
            Line2D::from_points(p0, p1)
        };

        let mut x = Vec::new();

        for i in 0..sz {
            let p0 = &self[i];
            let p1 = &self[(i + 1) % sz];
            let seg = LineSegment2D::from_points(*p0, *p1);
            
            if let Some(q) = seg.intersect_line(&line) {
                x.push(q.x());
            }
        }

        x.sort_unstable_by(f64::total_cmp);
        x
    }
}


// Compute the crossings of the polygon along line y, storing them in
// the array x. X must be allocated to be at least as long as
// poly.len(). X will be sorted, ready for
// rasterization. Returns the number of intersections (and elements
// written to x).
/*
  To rasterize, do something like this:

  double res = 0.099;
  for (double y = y0; y < y1; y += res) {
  double xs[poly.len()];

  int xsz = g2d_polygon_rasterize(poly, y, xs);
  int xpos = 0;
  int inout = 0; // start off "out"

  for (double x = x0; x < x1; x += res) {
      while (x > xs[xpos] && xpos < xsz) {
        xpos++;
        inout ^= 1;
      }

    if (inout)
       printf("y");
    else
       printf(" ");
  }
  printf("\n");
*/

mod bench {
    /*
    /*
  /---(1,5)
  (-2,4)-/        |
  \          |
  \        (1,2)--(2,2)\
  \                     \
  \                      \
  (0,0)------------------(4,0)
*/
#if 0

#include "timeprofile.h"

int main(int argc, char *argv[])
{
    timeprofile_t *tp = timeprofile_create();

    zarray_t *polya = g2d_polygon_create_data((double[][2]) {
            { 0, 0},
            { 4, 0},
            { 2, 2},
            { 1, 2},
            { 1, 5},
            { -2,4} }, 6);

    zarray_t *polyb = g2d_polygon_create_data((double[][2]) {
            { .1, .1},
            { .5, .1},
            { .1, .5 } }, 3);

    zarray_t *polyc = g2d_polygon_create_data((double[][2]) {
            { 3, 0},
            { 5, 0},
            { 5, 1} }, 3);

    zarray_t *polyd = g2d_polygon_create_data((double[][2]) {
            { 5, 5},
            { 6, 6},
            { 5, 6} }, 3);

/*
  5      L---K
  4      |I--J
  3      |H-G
  2      |E-F
  1      |D--C
  0      A---B
  01234
*/
    zarray_t *polyE = g2d_polygon_create_data((double[][2]) {
            {0,0}, {4,0}, {4, 1}, {1,1},
                                  {1,2}, {3,2}, {3,3}, {1,3},
                                                       {1,4}, {4,4}, {4,5}, {0,5}}, 12);

    srand(0);

    timeprofile_stamp(tp, "begin");

    if true {
        int niters = 100000;

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            g2d_polygon_contains_point(polyE, q);
        }

        timeprofile_stamp(tp, "fast");

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            g2d_polygon_contains_point_ref(polyE, q);
        }

        timeprofile_stamp(tp, "slow");

        for (int i = 0; i < niters; i++) {
            double q[2];
            q[0] = 10.0f * random() / RAND_MAX - 2;
            q[1] = 10.0f * random() / RAND_MAX - 2;

            int v0 = g2d_polygon_contains_point(polyE, q);
            int v1 = g2d_polygon_contains_point_ref(polyE, q);
            assert(v0 == v1);
        }

        timeprofile_stamp(tp, "both");
        timeprofile_display(tp);
    }

    if true {
        zarray_t *poly = polyE;

        double res = 0.399;
        for (double y = 5.2; y >= -.5; y -= res) {
            double xs[poly.len()];

            int xsz = g2d_polygon_rasterize(poly, y, xs);
            int xpos = 0;
            int inout = 0; // start off "out"
            for (double x = -3; x < 6; x += res) {
                while (x > xs[xpos] && xpos < xsz) {
                    xpos++;
                    inout ^= 1;
                }

                if (inout)
                    printf("y");
                else
                    printf(" ");
            }
            printf("\n");

            for (double x = -3; x < 6; x += res) {
                double q[2] = {x, y};
                if (g2d_polygon_contains_point(poly, q))
                    printf("X");
                else
                    printf(" ");
            }
            printf("\n");
        }
    }



/*
// CW order
double p[][2] =  { { 0, 0},
{ -2, 4},
{1, 5},
{1, 2},
{2, 2},
{4, 0} };
*/

     double q[2] = { 10, 10 };
     printf("0==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 1; q[1] = 1;
     printf("1==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 3; q[1] = .5;
     printf("1==%d\n", g2d_polygon_contains_point(polya, q));

     q[0] = 1.2; q[1] = 2.1;
     printf("0==%d\n", g2d_polygon_contains_point(polya, q));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyb));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyc));

     printf("0==%d\n", g2d_polygon_contains_polygon(polya, polyd));

     ////////////////////////////////////////////////////////
     // Test convex hull
     if true {
         zarray_t *hull = g2d_convex_hull(polyE);

         for (int k = 0; k < zarray_size(hull); k++) {
             double *h;
             zarray_get_volatile(hull, k, &h);

             printf("%15f, %15f\n", h[0], h[1]);
         }
     }

     for (int i = 0; i < 100000; i++) {
         zarray_t *points = zarray_create(sizeof(double[2]));

         for (int j = 0; j < 100; j++) {
             double q[2];
             q[0] = 10.0f * random() / RAND_MAX - 2;
             q[1] = 10.0f * random() / RAND_MAX - 2;

             zarray_add(points, q);
         }

         zarray_t *hull = g2d_convex_hull(points);
         for (int j = 0; j < zarray_size(points); j++) {
             double *q;
             zarray_get_volatile(points, j, &q);

             int on_edge;

             double p[2];
             g2d_polygon_closest_boundary_point(hull, q, p);
             if (g2d_distance(q, p) < .00001)
                 on_edge = 1;

             assert(on_edge || g2d_polygon_contains_point(hull, q));
         }

         zarray_destroy(hull);
         zarray_destroy(points);
     }
}
#endif
 */
}