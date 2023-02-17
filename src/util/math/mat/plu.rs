use crate::util::mem::calloc;

use super::mat::Mat;

// All square matrices (even singular ones) have a partially-pivoted
// LU decomposition such that A = PLU, where P is a permutation
// matrix, L is a lower triangular matrix, and U is an upper
// triangular matrix.
//
pub(crate) struct MatPLU {
    // was the input matrix singular? When a zero pivot is found, this
    // flag is set to indicate that this has happened.
    pub(super) singular: bool,

    /// permutation indices
    piv: Box<[u32]>,
    /// either +1 or -1
    pub(super) pivsign: i32,

    // The matd_plu_t object returned "owns" the enclosed LU matrix. It
    // is not expected that the returned object is itself useful to
    // users: it contains the L and U information all smushed
    // together.
    /// combined L and U matrices, permuted so they can be triangular.
    lu: Mat,
}

impl MatPLU {
    pub(super) fn new(a: &Mat) -> Self {
        let piv = calloc::<u32>(a.rows());
        let mut pivsign = 1;
        let mut singular = false;

        let mut lu = a.clone();
    
        // only for square matrices.
        assert!(a.dims.is_square());
    
        for i in 0..a.rows() {
            piv[i] = i as u32;
        }
    
        for j in 0..a.cols() {
            for i in 0..a.rows() {
                let kmax = std::cmp::min(i, j);
    
                // compute dot product of row i with column j (up through element kmax)
                let mut acc = 0.;
                for k in 0..kmax {
                    acc += lu[(i,k)] * lu[(k,j)];
                }
    
                lu[(i,j)] -= acc;
            }
    
            // find pivot and exchange if necessary.
            let mut p = j;
            if true {
                for i in (j+1)..lu.rows() {
                    if lu[(i,j)].abs() > lu[(p,j)].abs() {
                        p = i;
                    }
                }
            }
    
            // swap rows p and j?
            if p != j {
                assert!(p > j); // I'm pretty sure this is true. If this assert fails, we'll have to add some more code in the row swapping part

                let (row_j, row_p) = {
                    let (left, right) = lu.data.split_at_mut(p * lu.cols());
                    let j_start = j * lu.cols();
                    let row_j = &mut left[j_start..j_start + lu.cols()];
                    let row_p = &mut right[0..lu.cols()];
                    (row_j, row_p)
                };
                row_j.swap_with_slice(row_p);
                piv.swap(p, j);
                pivsign = -pivsign;
            }
    
            let LUjj = lu[(j,j)];
    
            // If our pivot is very small (which means the matrix is
            // singular or nearly singular), replace with a new pivot of the
            // right sign.
            if LUjj.abs() < Mat::EPS {
    /*
                if (LUjj < 0)
                    LUjj = -MATD_EPS;
                else
                    LUjj = MATD_EPS;
    
                MATD_EL(lu, j, j) = LUjj;
    */
                singular = true;
            }
    
            if j < lu.cols() && j < lu.rows() && LUjj != 0. {
                LUjj = LUjj.recip();
                for i in (j+1)..lu.rows() {
                    lu[(i,j)] *= LUjj;
                }
            }
        }

        Self {
            lu,
            piv,
            pivsign,
            singular,
        }
    }

    pub fn det(&self) -> f64 {
        let mut det = self.pivsign as f64;

        if self.lu.dims.is_square() {
            for i in 0..self.lu.cols() {
                det *= self.lu[(i,i)];
            }
        }

        det
    }

    pub fn p(&self) -> Mat {
        let ref lu = self.lu;
        let P = Mat::zeroes(lu.rows(), lu.rows());
    
        for i in 0..lu.rows() {
            P[(self.piv[i] as usize, i)] = 1.;
        }
    
        P
    }

    pub fn lower(&self) -> Mat {
        let lu = &self.lu;

        let L = Mat::zeroes_like(&lu);
        for i in 0..lu.rows() {
            L[(i,i)] = 1.;
            for j in 0..i {
                L[(i,j)] = lu[(i,j)];
            }
        }

        L
    }

    pub fn upper(&self) -> Mat {
        let lu = &self.lu;

        let U = Mat::zeroes(lu.cols(), lu.cols());
        for i in 0..lu.cols() {
            for j in 0..lu.cols() {
                if i <= j {
                    U[(i,j)] = lu[(i,j)];
                }
            }
        }
    
        U
    }

    pub fn solve(&self, b: &Mat) -> Mat {
        let mut x = b.clone();

        // permute right hand side
        for i in 0..self.lu.rows() {
            let xstart = i * x.cols();
            let bstart = self.piv[i] as usize;
            x.data[xstart..xstart+b.cols()].copy_from_slice(&b.data[bstart..bstart+b.cols()]);
        }
    
        // solve Ly = b
        for k in 0..self.lu.rows() {
            for i in (k+1)..self.lu.rows() {
                let LUik = -self.lu[(i,k)];
                for t in 0..b.cols() {
                    x[(i,t)] += x[(k,t)] * LUik;
                }
            }
        }
    
        // solve Ux = y
        for k in (0..self.lu.cols()).rev() {
            let LUkk = self.lu[(k,k)].recip();
            for t in 0..b.cols() {
                x[(k,t)] *= LUkk;
            }
    
            for i in 0..k {
                let LUik = -self.lu[(i,k)];
                for t in 0..b.cols() {
                    x[(i,t)] += x[(k,t)] * LUik;
                }
            }
        }
    
        x
    }
}