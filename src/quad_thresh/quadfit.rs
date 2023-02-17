use std::collections::BinaryHeap;
use rayon::prelude::*;

use crate::{util::{Image, mem::calloc}, ApriltagDetector, quad_decode::Quad};

use super::{linefit::{Pt, LineFitPoint, fit_line, ptsort, self, LineFitData}};

struct Segment {
    // always greater than zero, but right can be > size, which denotes
    // a wrap around back to the beginning of the points. and left < right.
    left: usize,
    right: usize,

    is_vertex: bool,
}

struct RemoveVertex {
    /// which vertex to remove?
    i: usize,
    /// left vertex
    left: usize,
    /// right vertex
    right: usize,
    err: f64,
}

// Implement ordering for MaxHeap
impl PartialEq for RemoveVertex {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.left == other.left && self.right == other.right && self.err == other.err
    }
}

impl Eq for RemoveVertex {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialOrd for RemoveVertex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(f64::partial_cmp(&self.err, &other.err)?.reverse())
    }
}

impl Ord for RemoveVertex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        return f64::total_cmp(&self.err, &other.err).reverse()
    }
}

/*

  1. Identify A) white points near a black point and B) black points near a white point.

  2. Find the connected components within each of the classes above,
  yielding clusters of "white-near-black" and
  "black-near-white". (These two classes are kept separate). Each
  segment has a unique id.

  3. For every pair of "white-near-black" and "black-near-white"
  clusters, find the set of points that are in one and adjacent to the
  other. In other words, a "boundary" layer between the two
  clusters. (This is actually performed by iterating over the pixels,
  rather than pairs of clusters.) Critically, this helps keep nearby
  edges from becoming connected.
*/
fn quad_segment_maxima(td: &ApriltagDetector, cluster: &[Pt], lfps: &[LineFitPoint]) -> Option<[usize; 4]> {
    let sz = cluster.len();

    // ksz: when fitting points, how many points on either side do we consider?
    // (actual "kernel" width is 2ksz).
    //
    // This value should be about: 0.5 * (points along shortest edge).
    //
    // If all edges were equally-sized, that would give a value of
    // sz/8. We make it somewhat smaller to account for tags at high
    // aspects.

    // XXX Tunable. Maybe make a multiple of JPEG block size to increase robustness
    // to JPEG compression artifacts?
    let ksz = std::cmp::min(20, sz / 12);

    // can't fit a quad if there are too few points.
    if ksz < 2 {
        return 0;
    }

//    printf("sz %5d, ksz %3d\n", sz, ksz);

    let mut errs = calloc::<f64>(sz);
    for i in 0..sz {
        errs[i] = fit_line(lfps, (i + sz - ksz) % sz, (i + ksz) % sz).err;
    }

    // apply a low-pass filter to errs
    if true {
        // how much filter to apply?

        // XXX Tunable
        let sigma = 1.; // was 3

        // cutoff = exp(-j*j/(2*sigma*sigma));
        // log(cutoff) = -j*j / (2*sigma*sigma)
        // log(cutoff)*2*sigma*sigma = -j*j;

        // how big a filter should we use? We make our kernel big
        // enough such that we represent any values larger than
        // 'cutoff'.

        // XXX Tunable (though not super useful to change)
        let cutoff = 0.05;
        let fsz = f64::sqrt(-f64::ln(cutoff) * 2. * sigma * sigma) as usize + 1;
        let fsz = 2*fsz + 1;

        // For default values of cutoff = 0.05, sigma = 3,
        // we have fsz = 17.
        let mut f = calloc::<f32>(fsz);
        for i in 0..fsz {
            let j = (i - fsz / 2) as f32;
            f[i] = f32::exp(-j*j/(2.*(sigma*sigma) as f32));
        }

        let mut y = calloc::<f64>(sz);
        for iy in 0..sz {
            let mut acc = 0.;

            for i in 0..fsz {
                acc += errs[(iy + i - fsz / 2 + sz) % sz] * (f[i] as f64);
            }
            y[iy] = acc;
        }

        errs.copy_from_slice(&y);
    }

    let mut maxima = Vec::<(usize, f64)>::new();
    // let mut maxima: [usize; sz];
    // let mut maxima_errs: [f64; sz];
    // let mut nmaxima = 0;

    for i in 0..sz {
        if errs[i] > errs[(i+1)%sz] && errs[i] > errs[(i+sz-1)%sz] {
            maxima.push((i, errs[i]));
        }
    }

    // if we didn't get at least 4 maxima, we can't fit a quad.
    if maxima.len() < 4 {
        return 0;
    }

    // select only the best maxima if we have too many
    let max_nmaxima = td.qtp.max_nmaxima;

    if maxima.len() > (max_nmaxima as usize) {
        let mut maxima_errs_copy = maxima
            .iter()
            .map(|(_maximum, err)| *err)
            .collect::<Vec<_>>();

        // throw out all but the best handful of maxima. Sorts descending.
        maxima_errs_copy.sort_by(|a, b| f64::total_cmp(a, b).reverse());

        let maxima_thresh = maxima_errs_copy[max_nmaxima as usize];
        let mut out = 0;

        maxima = maxima
            .into_iter()
            .filter(|(_maximum, err)| *err > maxima_thresh)
            .collect::<Vec<_>>();
    }

    let mut best_indices: [usize; 4];
    let mut best_error = f64::INFINITY;

    // disallow quads where the angle is less than a critical value.
    let max_dot = td.qtp.cos_critical_rad; //25*M_PI/180);

    for m0 in 0..(maxima.len() - 3) {
        let (i0, _i0_err) = maxima[m0];

        for m1 in (m0+1)..(maxima.len() - 2) {
            let (i1, _i1_err) = maxima[m1];

            let line01 = fit_line(lfps, i0, i1);

            if line01.mse > td.qtp.max_line_fit_mse as f64 {
                continue;
            }

            for m2 in (m1+1)..(maxima.len() - 1) {
                let (i2, _i2_err) = maxima[m2];

                let line12 = fit_line(lfps, i1, i2);
                if line12.mse > td.qtp.max_line_fit_mse as f64 {
                    continue;
                }

                let dot = line01.lineparm[2]*line12.lineparm[2] + line01.lineparm[3]*line12.lineparm[3];
                if dot.abs() > max_dot as f64 {
                    continue;
                }

                for m3 in (m2+1)..maxima.len() {
                    let (i3, _i3_err) = maxima[m3];

                    let line23 = fit_line(lfps, i2, i3);
                    if line23.mse > td.qtp.max_line_fit_mse as f64 {
                        continue;
                    }

                    let line30 = fit_line(lfps, i3, i0);
                    if line30.mse > td.qtp.max_line_fit_mse as f64 {
                        continue;
                    }

                    let err = line01.err + line12.err + line23.err + line30.err;
                    if err < best_error {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = i1;
                        best_indices[2] = i2;
                        best_indices[3] = i3;
                    }
                }
            }
        }
    }

    if best_error == f64::INFINITY {
        return None;
    }

    if best_error / (sz as f64) >= td.qtp.max_line_fit_mse as f64 {
        return None;
    }
    Some(best_indices)
}

/// returns false if the cluster looks bad.
fn quad_segment_agg(td: &ApriltagDetector, cluster: &[Pt], lfps: &[LineFitPoint]) -> Option<[usize; 4]> {
    let sz = cluster.len();

    let mut heap: BinaryHeap<RemoveVertex> = BinaryHeap::new();

    // We will initially allocate sz rvs. We then have two types of
    // iterations: some iterations that are no-ops in terms of
    // allocations, and those that remove a vertex and allocate two
    // more children.  This will happen at most (sz-4) times.  Thus we
    // need: sz + 2*(sz-4) entries.

    // let mut rvalloc_pos = 0;
    // let rvalloc_size = 3*sz;
    let segs = Vec::<Segment>::new();

    // populate with initial entries
    for i in 0..cluster.len() {
        let (left, right) = if i == 0 {
            (cluster.len() - 1, 1)
        } else {
            (i - 1, (i + 1) % cluster.len())
        };

        let LineFitData { err, .. } = linefit::fit_line(&lfps, left, right);

        let rv = RemoveVertex { i, left, right, err };

        heap.push(rv);

        segs.push(Segment {
            left: rv.left,
            right: rv.right,
            is_vertex: true,
        });
    }

    let mut nvertices = sz;
    while nvertices > 4 {
        assert!(rvalloc_pos < rvalloc_size);

        let rv = heap.pop()?;

        // is this remove_vertex valid? (Or has one of the left/right
        // vertices changes since we last looked?)
        if !segs[rv.i].is_vertex ||
            !segs[rv.left].is_vertex ||
            !segs[rv.right].is_vertex {
            continue;
        }

        // we now merge.
        assert!(segs[rv.i].is_vertex);

        segs[rv.i].is_vertex = false;
        segs[rv.left].right = rv.right;
        segs[rv.right].left = rv.left;

        // create the join to the left
        if true {
            let i = rv.left;
            let left = segs[rv.left].left;
            let right = rv.right;

            let LineFitData { err, .. } = linefit::fit_line(&lfps, left, right);

            heap.push(RemoveVertex { i, left, right, err });
        }

        // create the join to the right
        if true {
            let i = rv.right;
            let left = rv.left;
            let right = segs[rv.right].right;

            let LineFitData { err, .. } = linefit::fit_line(&lfps, left, right);

            heap.push(RemoveVertex { i, left, right, err });
        }

        // we now have one less vertex
        nvertices -= 1;
    }

    let indices = segs
        .into_iter()
        .enumerate()
        .filter(|(_idx, seg)| seg.is_vertex)
        .map(|(idx, _seg)| idx)
        .collect::<Vec<_>>();

    Some(indices)
}

/// Compute statistics that allow line fit queries to be
/// efficiently computed for any contiguous range of indices.
fn compute_lfps(cluster: &[Pt], im: &Image) -> Vec<LineFitPoint> {
    let mut lfps = Vec::with_capacity(cluster.len());

    let default = LineFitPoint::default();

    for p in cluster {
        let last = lfps.last().unwrap_or(&default);

        {
            // we now undo our fixed-point arithmetic.
            let delta = 0.5; // adjust for pixel center bias
            let x = p.x as f64 * 0.5 + delta;
            let y = p.y as f64 * 0.5 + delta;
            let ix = x as usize;
            let iy = y as usize;
            let mut W = 1.;

            if ix > 0 && ix+1 < im.width && iy > 0 && iy+1 < im.height {
                let grad_x = (im[(ix + 1, iy)] - im[(ix - 1, iy)]) as f64;
                let grad_y = (im[(ix, iy + 1)] - im[(ix, iy - 1)]) as f64;

                // XXX Tunable. How to shape the gradient magnitude?
                W = (grad_x*grad_x + grad_y*grad_y).sqrt() + 1.;
            }

            let fx = x;
            let fy = y;
            lfps.push(LineFitPoint {
                Mx:  last.Mx  + W * fx,
                My:  last.My  + W * fy,
                Mxx: last.Mxx + W * fx * fx,
                Myy: last.Myy + W * fx * fy,
                Mxy: last.Mxy + W * fy * fy,
                W: last.W + W,
            });
        }
    }
    lfps
}

// return 1 if the quad looks okay, 0 if it should be discarded
fn fit_quad(td: ApriltagDetector, im: &Image, cluster: &mut [Pt], tag_width: usize, normal_border: bool, reversed_border: bool) -> Option<Quad> {
    let sz = cluster.len();
    if sz < 24 {// Synchronize with later check.
        return None;
    }

    let mut res = false;

    /////////////////////////////////////////////////////////////
    // Step 1. Sort points so they wrap around the center of the
    // quad. We will constrain our quad fit to simply partition this
    // ordered set into 4 groups.

    // compute a bounding box so that we can order the points
    // according to their angle WRT the center.
    let p1 = &cluster[0];
    let mut xmax = p1.x;
    let mut xmin = p1.x;
    let mut ymax = p1.y;
    let mut ymin = p1.y;
    for p in cluster.iter().skip(1) {
        if p.x > xmax {
            xmax = p.x;
        } else if p.x < xmin {
            xmin = p.x;
        }

        if p.y > ymax {
            ymax = p.y;
        } else if p.y < ymin {
            ymin = p.y;
        }
    }

    if (xmax - xmin)*(ymax - ymin) < tag_width {
        return None;
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    let cx: f32 = (xmin + xmax) * 0.5 + 0.05118;
    let cy: f32 = (ymin + ymax) * 0.5 + -0.028581;

    let quadrants: [[f32; 2]; 2] = [
        [-1.*(2 << 15) as f32, 0.],
        [ 2.*(2 << 15) as f32, (2 << 15) as f32],
    ];

    let mut dot = 0.;
    for p in cluster.iter_mut() {
        let dx = p.x as f32 - cx;
        let dy = p.y as f32 - cy;

        dot += dx*p.gx as f32 + dy*p.gy as f32;

        let quadrant = quadrants[dy > 0][dx > 0];
        let (dx, dy) = if dy < 0 {
            (-dx, -dy)
        } else {
            (dx, dy)
        };

        let (dx, dy) = if dx < 0 {
            (dy, -dx)
        } else {
            (dx, dy)
        };

        p.slope = quadrant + dy/dx;
    }

    // Ensure that the black border is inside the white border.
    quad.reversed_border = dot < 0.;
    if !reversed_border && quad.reversed_border {
        return None;
    }
    if !normal_border && !quad.reversed_border {
        return None;
    }

    // we now sort the points according to theta. This is a prepatory
    // step for segmenting them into four lines.
    if true {
        ptsort(&mut cluster);

        // remove duplicate points. (A byproduct of our segmentation system.)
        if true {
            let mut outpos = 1;

            let mut last = &cluster[0];
            for p in cluster.iter().skip(1) {
                if p.x != last.x || p.y != last.y {
                    if i != outpos  {
                        struct pt *out;
                        zarray_get_volatile(cluster, outpos, &out);
                        memcpy(out, p, sizeof(struct pt));
                    }

                    outpos++;
                }

                last = p;
            }

            cluster.size = outpos;
            sz = outpos;
        }

    }

    if sz < 24 {
        return None;
    }

    let lfps = compute_lfps(cluster, im);

    int indices[4];
    if true {
        if (!quad_segment_maxima(td, cluster, lfps, indices))
            goto finish;
    } else {
        if (!quad_segment_agg(td, cluster, lfps, indices))
            goto finish;
    }


    let mut lines: [[f64; 4]; 4];
    for i in 0..4 {
        let i0 = indices[i];
        let i1 = indices[(i+1)&3];

        let line = fit_line(&lfps, i0, i1);
        lines[i] = line.lineparm;

        if (line.err > td.qtp.max_line_fit_mse) {
            return None;
        }
    }

    for i in 0..4 {
        // solve for the intersection of lines (i) and (i+1)&3.
        // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
        // are the line directions.
        //
        // lambda0*u0 - lambda1*u1 = (p1 - p0)
        //
        // rearrange (solve for lambdas)
        //
        // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
        // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
        //
        // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
        // We want the unit vector, so we need the perpendiculars. Thus, below
        // we have swapped the x and y components and flipped the y components.

        let A00 =  lines[i][3];
        let A01 = -lines[(i+1)&3][3];
        let A10 =  -lines[i][2];
        let A11 = lines[(i+1)&3][2];
        let B0 = -lines[i][0] + lines[(i+1)&3][0];
        let B1 = -lines[i][1] + lines[(i+1)&3][1];

        let det = A00 * A11 - A10 * A01;

        // inverse.
        let W00 = A11 / det;
        let W01 = -A01 / det;
        if (det.abs() < 0.001) {
            return None;
        }

        // solve
        let L0 = W00*B0 + W01*B1;

        // compute intersection
        quad.corners[i][0] = lines[i][0] + L0*A00;
        quad.corners[i][1] = lines[i][1] + L0*A10;

        res = 1;
    }

    // reject quads that are too small
    if true {
        let mut area = 0.;

        // get area of triangle formed by points 0, 1, 2, 0
        let mut length: [f64; 3];
        for i in 0..3 {
            let idxa = i; // 0, 1, 2,
            let idxb = (i+1) % 3; // 1, 2, 0
            length[i] = (sq(quad.corners[idxb][0] - quad.corners[idxa][0]) +
                            sq(quad.corners[idxb][1] - quad.corners[idxa][1])).sqrt();
        }
        let p = (length[0] + length[1] + length[2]) / 2.;

        area += (p*(p-length[0])*(p-length[1])*(p-length[2])).sqrt();

        // get area of triangle formed by points 2, 3, 0, 2
        for i in 0..3 {
            let idxs = [ 2, 3, 0, 2 ];
            let idxa = idxs[i];
            let idxb = idxs[i+1];
            length[i] = (sq(quad.corners[idxb][0] - quad.corners[idxa][0]) +
                            sq(quad.corners[idxb][1] - quad.corners[idxa][1])).sqrt();
        }
        let p = (length[0] + length[1] + length[2]) / 2.;

        area += (p*(p-length[0])*(p-length[1])*(p-length[2])).sqrt();

        if (area < tag_width*tag_width) {
            return None;
        }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    if true {
        for i in 0..4 {
            let i0 = i;
            let i1 = (i+1)&3;
            let i2 = (i+2)&3;

            let dx1 = quad.corners[i1][0] - quad.corners[i0][0];
            let dy1 = quad.corners[i1][1] - quad.corners[i0][1];
            let dx2 = quad.corners[i2][0] - quad.corners[i1][0];
            let dy2 = quad.corners[i2][1] - quad.corners[i1][1];
            let cos_dtheta = (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));

            if ((cos_dtheta > td.qtp.cos_critical_rad || cos_dtheta < -td.qtp.cos_critical_rad) || dx1*dy2 < dy1*dx2) {
                return None;
            }
        }
    }
    res
}
/*
// return true if the quad looks okay, false if it should be discarded
fn fit_quad(td: &ApriltagDetector, im: &Image, cluster: &mut [pt], tag_width: usize, normal_border: bool, reversed_border: bool) -> Option<Quad> {
    // can't fit a quad to less than 4 points
    if cluster.len() < 4 {
        return None;
    }

    /////////////////////////////////////////////////////////////
    // Step 1. Sort points so they wrap around the center of the
    // quad. We will constrain our quad fit to simply partition this
    // ordered set into 4 groups.

    // compute a bounding box so that we can order the points
    // according to their angle WRT the center.
    let mut xmax = 0;
    let mut xmin = i32::MAX;
    let mut ymax = 0;
    let mut ymin = i32::MAX;

    for p in cluster {
        xmax = i32::max(xmax, p.x as i32);
        xmin = i32::min(xmin, p.x as i32);

        ymax = i32::max(ymax, p.y as i32);
        ymin = i32::min(ymin, p.y as i32);
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    let cx = (xmin + xmax) as f32 * 0.5 + 0.05118;
    let cy = (ymin + ymax) as f32 * 0.5 + -0.028581;

    let dot = 0.;

    for p in cluster {
        let dx = p.x as f32 - cx;
        let dy = p.y as f32 - cy;

        p.theta = f32::atan2(dy, dx);

        dot += dx*(p.gx as f32) + dy*(p.gy as f32);
//        p.theta = terrible_atan2(dy, dx);
    }

    // Ensure that the black border is inside the white border.
    if dot < 0. {
        return false;
    }

    // we now sort the points according to theta. This is a prepatory
    // step for segmenting them into four lines.
    if true {
        //        zarray_sort(cluster, pt_compare_theta);
        ptsort(cluster);

        // remove duplicate points. (A byproduct of our segmentation system.)
        if true {
            let mut outpos = 1;
            let mut last = &cluster[0];

            for i in 1..cluster.len() {
                let p = &cluster[i];

                if p.x != last.x || p.y != last.y {
                    if i != outpos  {
                        cluster.swap(outpos, i);
                    }

                    outpos += 1;
                }

                last = p;
            }

            cluster.size = outpos;
            sz = outpos;
        }

    } else {
        // This is a counting sort in which we retain at most one
        // point for every bucket; the bucket index is computed from
        // theta. Since a good quad completes a complete revolution,
        // there's reason to think that we should get a good
        // distribution of thetas.  We might "lose" a few points due
        // to collisions, but this shouldn't affect quality very much.

        // XXX tunable. Increase to reduce the likelihood of "losing"
        // points due to collisions.
        let nbuckets = 4*sz;

        const ASSOC: usize = 2;
        let v = calloc::<[pt; 2]>(nbuckets);

        // put each point into a bucket.
        for p in cluster {
            assert!(p.theta >= -f32c::PI && p.theta <= f32c::PI);

            let bucket = (nbuckets - 1) as f32 * (p.theta + f32c::PI) / (2.*f32c::PI);
            assert!(bucket >= 0 && bucket < nbuckets);

            for i in 0..ASSOC {
                if v[bucket][i].theta == 0 {
                    v[bucket][i] = *p;
                    break;
                }
            }
        }

        // collect the points from the buckets and put them back into the array.
        let mut outsz = 0;
        for i in 0..nbuckets {
            for j in 0..ASSOC {
                if v[i][j].theta != 0 {
                    cluster[outsz] = &v[i][j];
                    outsz += 1;
                }
            }
        }

        zarray_truncate(cluster, outsz);
        sz = outsz;
    }

    if sz < 4 {
        return 0;
    }

    /////////////////////////////////////////////////////////////
    // Step 2. Precompute statistics that allow line fit queries to be
    // efficiently computed for any contiguous range of indices.

    let lfps = compute_lfps(cluster, im);
    // struct line_fit_pt *lfps = calloc(sz, sizeof(struct line_fit_pt));

    let mut indices: [i32; 4];
    if true {
        if quad_segment_maxima(td, cluster, &lfps, indices) == 0 {
            return false;
        }
    } else {
        if !quad_segment_agg(td, cluster, &lfps, indices) {
            return false;
        }
    }

//    printf("%d %d %d %d\n", indices[0], indices[1], indices[2], indices[3]);

    if false {
        // no refitting here; just use those points as the vertices.
        // Note, this is useful for debugging, but pretty bad in
        // practice since this code path also omits several
        // plausibility checks that save us tons of time in quad
        // decoding.
        for i in 0..4 {
            let p = &cluster[indices[i]];

            quad.corners[i][0] = 0.5*p.x(); // undo fixed-point arith.
            quad.corners[i][1] = 0.5*p.y();
        }

        res = 1;

    } else {
        let mut lines: [[f64; 4]; 4];

        for i in 0..4 {
            let i0 = indices[i];
            let i1 = indices[(i+1)&3];

            if false {
                // if there are enough points, skip the points near the corners
                // (because those tend not to be very good.)
                if (i1-i0 > 8) {
                    let mut t = (i1-i0)/6;
                    if (t < 0) {
                        t = -t;
                    }

                    i0 = (i0 + t) % sz;
                    i1 = (i1 + sz - t) % sz;
                }
            }

            let LineFitData { err, .. } = fit_line(&lfps, i0, i1);
            if (err > td.qtp.max_line_fit_mse) {
                res = 0;
                goto finish;
            }
        }

        for i in 0..4 {
            // solve for the intersection of lines (i) and (i+1)&3.
            // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
            // are the line directions.
            //
            // lambda0*u0 - lambda1*u1 = (p1 - p0)
            //
            // rearrange (solve for lambdas)
            //
            // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
            // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
            //
            // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
            // We want the unit vector, so we need the perpendiculars. Thus, below
            // we have swapped the x and y components and flipped the y components.

            let A00 =  lines[i][3];
            let A01 = -lines[(i+1)&3][3];
            let A10 =  -lines[i][2];
            let A11 = lines[(i+1)&3][2];
            let B0 = -lines[i][0] + lines[(i+1)&3][0];
            let B1 = -lines[i][1] + lines[(i+1)&3][1];

            let det = A00 * A11 - A10 * A01;

            // inverse.
            let W00 = A11 / det;
            let W01 = -A01 / det;
            if det.abs() < 0.001 {
                res = 0;
                goto finish;
            }

            // solve
            let L0 = W00*B0 + W01*B1;

            // compute intersection
            quad.corners[i][0] = lines[i][0] + L0*A00;
            quad.corners[i][1] = lines[i][1] + L0*A10;

            if false {
                // we should get the same intersection starting
                // from point p1 and moving L1*u1.
                let W10 = -A10 / det;
                let W11 = A00 / det;
                let L1 = W10*B0 + W11*B1;

                let x = lines[(i+1)&3][0] - L1*A10;
                let y = lines[(i+1)&3][1] - L1*A11;
                assert!((x - quad.corners[i].x()).abs() < 0.001);
                assert!((y - quad.corners[i].y()).abs() < 0.001);
            }

            res = 1;
        }
    }

    // reject quads that are too small
    if true {
        let mut area = 0;

        // get area of triangle formed by points 0, 1, 2, 0
        let length: [f64; 3];
        for i in 0..3 {
            let idxa = i; // 0, 1, 2,
            let idxb = (i+1) % 3; // 1, 2, 0
            length[i] = sqrt(sq(quad.corners[idxb][0] - quad.corners[idxa][0]) +
                             sq(quad.corners[idxb][1] - quad.corners[idxa][1]));
        }
        let p = (length[0] + length[1] + length[2]) / 2.;

        area += (p*(p-length[0])*(p-length[1])*(p-length[2])).sqrt();

        // get area of triangle formed by points 2, 3, 0, 2
        for i in 0..3 {
            let idxs: = [ 2, 3, 0, 2 ];
            let idxa = idxs[i];
            let idxb = idxs[i+1];
            length[i] = sqrt(sq(quad.corners[idxb][0] - quad.corners[idxa][0]) +
                             sq(quad.corners[idxb][1] - quad.corners[idxa][1]));
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += (p*(p-length[0])*(p-length[1])*(p-length[2])).sqrt();

        // we don't actually know the family yet (quad detection is generic.)
        // This threshold is based on a 6x6 tag (which is actually 8x8)
//        int d = fam.d + fam.black_border*2;
        int d = 8;
        if (area < d*d) {
            res = 0;
            goto finish;
        }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    if true {
        let mut total = 0.;

        for i in 0..4 {
            let i0 = i;
            let i1 = (i+1)&3;
            let i2 = (i+2)&3;

            let theta0 = f64::atan2(quad.corners[i0][1] - quad.corners[i1][1],
                                   quad.corners[i0][0] - quad.corners[i1][0]);
            let theta1 = atan2f(quad.corners[i2][1] - quad.corners[i1][1],
                                   quad.corners[i2][0] - quad.corners[i1][0]);

            let mut dtheta = theta0 - theta1;
            if dtheta < 0 {
                dtheta += 2*M_PI;
            }

            if (dtheta < td.qtp.critical_rad || dtheta > (M_PI - td.qtp.critical_rad))
                res = 0;

            total += dtheta;
        }

        // looking for 2PI
        if total < 6.2 || total > 6.4 {
            res = 0;
            goto finish;
        }
    }

    // adjust pixel coordinates; all math up 'til now uses pixel
    // coordinates in which (0,0) is the lower left corner. But each
    // pixel actually spans from to [x, x+1), [y, y+1) the mean value of which
    // is +.5 higher than x & y.
/*    double delta = .5;
      for i in 0..4 {
      quad.corners[i][0] += delta;
      quad.corners[i][1] += delta;
      }
*/
    return res;
}
*/
pub(super) fn fit_quads(td: &ApriltagDetector, w: usize, h: usize, clusters: &[Vec<Pt>], im: &Image) -> Vec<Quad> {
    let mut normal_border = false;
    let mut reversed_border = false;
    let mut min_tag_width = 1000000;
    for (ref family, _) in td.tag_families {
        min_tag_width = std::cmp::min(min_tag_width, family.width_at_border);

        normal_border |= !family.reversed_border;
        reversed_border |= family.reversed_border;
    }
    let min_tag_width = std::cmp::max(min_tag_width / td.params.quad_decimate, 3);

    // let chunksize = 1 + clusters.len() / (APRILTAG_TASKS_PER_THREAD_TARGET * td.params.nthreads);

    let quads = clusters
        .into_par_iter()
        .filter(|cluster| {
            if cluster.len() < td.qtp.min_cluster_pixels as usize {
                return false;
            }
            // a cluster should contain only boundary points around the
            // tag. it cannot be bigger than the whole screen. (Reject
            // large connected blobs that will be prohibitively slow to
            // fit quads to.) A typical point along an edge is added three
            // times (because it has 3 neighbors). The maximum perimeter
            // is 2w+2h.
            if cluster.len() > 3*(2*w+2*h) {
                return false;
            }
            true
        })
        .filter_map(|cluster| {
            fit_quad(td, im, &mut cluster, min_tag_width, normal_border, reversed_border)
        })
        .collect::<Vec<_>>();
    quads
}