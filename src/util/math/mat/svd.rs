use crate::util::mem::calloc;

use super::Mat;

pub(crate) struct MatSVD {
    pub(crate) U: Mat,
    pub(crate) S: Mat,
    pub(crate) V: Mat,
}

#[derive(Default)]
pub(crate) struct SvdOptions {
    pub suppress_warnings: bool,
}

const FIND_MAX_TOL: f64 = 1e-10;

trait FindMax {
    fn iter(&mut self, B: &Mat) -> Option<(usize, usize)>;
}

struct FindMaxReference {
}

impl FindMax for FindMaxReference {
    fn iter(&mut self, B: &Mat) -> Option<(usize, usize)> {
        // brute-force (reference) version.
        let mut maxv = -1.;

        let mut max_idxs = None;

        // only search top "square" portion
        for i in 0..B.cols() {
            for j in 0..B.cols() {
                if i == j {
                    continue;
                }

                let v = f64::abs(B[(i, j)]);

                if v > maxv {
                    max_idxs = Some((i, j));
                    maxv = v;
                }
            }
        }

        // termination condition.
        if maxv < FIND_MAX_TOL {
            None
        } else {
            Some(max_idxs.unwrap())
        }
    }
}

/// This method is the "smarter" method which does at least
/// 4*ncols work. More work might be needed (up to
/// ncols*ncols), depending on data. Thus, this might be a
/// bit slower than the default method for very small
/// matrices.
struct FindMaxFast {
    maxrowidx: Box<[usize]>,
    lastmaxi: usize,
    lastmaxj: usize,
}

impl FindMaxFast {
    fn new(B: &Mat) -> Self {
        // for each of the first B.cols() rows, which index has the
        // maximum absolute value? (used by method 1)
        let mut maxrowidx = calloc::<usize>(B.cols());
        for i in 2..B.cols() {
            maxrowidx[i] = B.max_idx(i, B.cols()).unwrap();
        }

        // note that we started the array at 2. That's because by setting
        // these values below, we'll recompute first two entries on the
        // first iteration!
        let lastmaxi = 0;
        let lastmaxj = 1;

        Self {
            maxrowidx,
            lastmaxi,
            lastmaxj,
        }
    }
}

impl FindMax for FindMaxFast {
    fn iter(&mut self, B: &Mat) -> Option<(usize, usize)> {
        let mut maxi: Option<usize> = None;
        let mut maxv = -1.;

        // every iteration, we must deal with the fact that rows
        // and columns lastmaxi and lastmaxj have been
        // modified. Update maxrowidx accordingly.

        // now, EVERY row also had columns lastmaxi and lastmaxj modified.
        for rowi in 0..B.cols() {

            // the magnitude of the largest off-diagonal element
            // in this row.
            let mut thismaxv: f64;

            // row 'lastmaxi' and 'lastmaxj' have been completely
            // changed. compute from scratch.
            if rowi == self.lastmaxi || rowi == self.lastmaxj {
                self.maxrowidx[rowi] = B.max_idx(rowi, B.cols()).unwrap();
                thismaxv = B[(rowi, self.maxrowidx[rowi])].abs();
                if thismaxv > maxv {
                    maxv = thismaxv;
                    maxi = Some(rowi);
                }
                continue;
            }

            // our maximum entry was just modified. We don't know
            // if it went up or down, and so we don't know if it
            // is still the maximum. We have to update from
            // scratch.
            if self.maxrowidx[rowi] == self.lastmaxi || self.maxrowidx[rowi] == self.lastmaxj {
                self.maxrowidx[rowi] = B.max_idx(rowi, B.cols()).unwrap();
                thismaxv = B[(rowi, self.maxrowidx[rowi])].abs();
                if thismaxv > maxv {
                    maxv = thismaxv;
                    maxi = Some(rowi);
                }
                continue;
            }

            // This row is unchanged, except for columns
            // 'lastmaxi' and 'lastmaxj', and those columns were
            // not previously the largest entry...  just check to
            // see if they are now the maximum entry in their
            // row. (Remembering to consider off-diagonal entries
            // only!)
            thismaxv = f64::abs(B[(rowi, self.maxrowidx[rowi])]);

            // check column lastmaxi. Is it now the maximum?
            if self.lastmaxi != rowi {
                let v = f64::abs(B[(rowi, self.lastmaxi)]);
                if v > thismaxv {
                    thismaxv = v;
                    self.maxrowidx[rowi] = self.lastmaxi;
                }
            }

            // check column lastmaxj
            if self.lastmaxj != rowi {
                let v = (B[(rowi, self.lastmaxj)]).abs();
                if v > thismaxv {
                    thismaxv = v;
                    self.maxrowidx[rowi] = self.lastmaxj;
                }
            }

            // does this row have the largest value we've seen so far?
            if thismaxv > maxv {
                maxv = thismaxv;
                maxi = Some(rowi);
            }
        }
        let maxi = maxi.unwrap();
        let maxj = self.maxrowidx[maxi];

        // save these for the next iteration.
        self.lastmaxi = maxi;
        self.lastmaxj = maxj;

        if maxv < FIND_MAX_TOL {
            None
        } else {
            Some((maxi, maxj))
        }
    }
}

/// SVD 2x2.
/// 
/// Computes singular values and vectors without squaring the input
/// matrix. With double precision math, results are accurate to about
/// 1E-16.
/// 
/// ```text
/// U = [ cos(theta) -sin(theta) ]
///     [ sin(theta)  cos(theta) ]
/// 
/// S = [ e  0 ]
///     [ 0  f ]
/// 
/// V = [ cos(phi)   -sin(phi) ]
///     [ sin(phi)   cos(phi)  ]
/// ```
/// 
/// 
/// Our strategy is basically to analytically multiply everything out
/// and then rearrange so that we can solve for theta, phi, e, and
/// f. (Derivation by ebolson@umich.edu 5/2016)
/// 
/// ```text
/// V' = [ CP  SP ]
///      [ -SP CP ]

/// USV' = [ CT -ST ][  e*CP  e*SP ]
///        [ ST  CT ][ -f*SP  f*CP ]
/// 
///      = [e*CT*CP + f*ST*SP     e*CT*SP - f*ST*CP ]
///        [e*ST*CP - f*SP*CT     e*SP*ST + f*CP*CT ]

/// A00+A11 = e*CT*CP + f*ST*SP + e*SP*ST + f*CP*CT
///         = e*(CP*CT + SP*ST) + f*(SP*ST + CP*CT)
///     	= (e+f)(CP*CT + SP*ST)
/// B0	    = (e+f)*cos(P-T)
/// 
/// A00-A11 = e*CT*CP + f*ST*SP - e*SP*ST - f*CP*CT
///         = e*(CP*CT - SP*ST) - f*(-ST*SP + CP*CT)
/// 	    = (e-f)(CP*CT - SP*ST)
/// B1	    = (e-f)*cos(P+T)
/// 
/// A01+A10 = e*CT*SP - f*ST*CP + e*ST*CP - f*SP*CT
/// 	    = e(CT*SP + ST*CP) - f*(ST*CP + SP*CT)
/// 	    = (e-f)*(CT*SP + ST*CP)
/// B2	    = (e-f)*sin(P+T)
/// 
/// A01-A10 = e*CT*SP - f*ST*CP - e*ST*CP + f*SP*CT
/// 	= e*(CT*SP - ST*CP) + f(SP*CT - ST*CP)
/// 	= (e+f)*(CT*SP - ST*CP)
/// B3	= (e+f)*sin(P-T)
/// 
/// B0 = (e+f)*cos(P-T)
/// B1 = (e-f)*cos(P+T)
/// B2 = (e-f)*sin(P+T)
/// B3 = (e+f)*sin(P-T)
/// 
/// B3/B0 = tan(P-T)
/// 
/// B2/B1 = tan(P+T)
/// ```
fn svd22(A: [f64; 4]) -> ([f64; 4], [f64; 2], [f64; 4]) {
    let A00 = A[0];
    let A01 = A[1];
    let A10 = A[2];
    let A11 = A[3];
 
    let B0 = A00 + A11;
    let B1 = A00 - A11;
    let B2 = A01 + A10;
    let B3 = A01 - A10;
 
    let PminusT = f64::atan2(B3, B0);
    let PplusT = f64::atan2(B2, B1);
 
    let P = (PminusT + PplusT) / 2.;
    let T = (-PminusT + PplusT) / 2.;
 
    let (SP, CP) = P.sin_cos();
    let (ST, CT) = T.sin_cos();
 
    let mut U = [
        CT,
        -ST,
        ST,
        CT,
    ];
 
    let mut V = [
        CP,
        -SP,
        SP,
        CP,
    ];
 
     // C0 = e+f. There are two ways to compute C0; we pick the one
     // that is better conditioned.
    let (SPmT, CPmT) = (P-T).sin_cos();
    let C0 = if CPmT.abs() > SPmT.abs() {
        B0 / CPmT
    } else {
        B3 / SPmT
    };
 
     // C1 = e-f. There are two ways to compute C1; we pick the one
     // that is better conditioned.
    let (SPpT, CPpT ) = (P+T).sin_cos();
    let C1 = if CPpT.abs() > SPpT.abs() {
        B1 / CPpT
    } else {
        B2 / SPpT
    };
 
     // e and f are the singular values
    let mut e = (C0 + C1) / 2.;
    let mut f = (C0 - C1) / 2.;
 
    if e < 0. {
        e = -e;
        U[0] = -U[0];
        U[2] = -U[2];
    }

    if f < 0. {
        f = -f;
        U[1] = -U[1];
        U[3] = -U[3];
    }
 
    // sort singular values.
    let S = if e > f {
        // already in big-to-small order.
        [e, f]
    } else {
        // Curiously, this code never seems to get invoked.  Why is it
        // that S[0] always ends up the dominant vector?  However,
        // this code has been tested (flipping the logic forces us to
        // sort the singular values in ascending order).
        //
        // P = [ 0 1 ; 1 0 ]
        // USV' = (UP)(PSP)(PV')
        //      = (UP)(PSP)(VP)'
        //      = (UP)(PSP)(P'V')'

        // exchange columns of U and V
        U.swap(0, 1);
        U.swap(2, 3);

        V.swap(0, 1);
        V.swap(2, 3);
        [f, e]
    };
 
     /*
     double SM[4] = { S[0], 0, 0, S[1] };
 
     doubles_print_mat(U, 2, 2, "%20.10g");
     doubles_print_mat(SM, 2, 2, "%20.10g");
     doubles_print_mat(V, 2, 2, "%20.10g");
     printf("A:\n");
     doubles_print_mat(A, 2, 2, "%20.10g");
 
     double SVt[4];
     doubles_mat_ABt(SM, 2, 2, V, 2, 2, SVt, 2, 2);
     double USVt[4];
     doubles_mat_AB(U, 2, 2, SVt, 2, 2, USVt, 2, 2);
 
     printf("USVt\n");
     doubles_print_mat(USVt, 2, 2, "%20.10g");
 
     double diff[4];
     for i in 0..4 {
         diff[i] = A[i] - USVt[i];
 
     printf("diff\n");
     doubles_print_mat(diff, 2, 2, "%20.10g");
 
     */

    (U, S, V)
}

impl MatSVD {
    pub(super) fn new(A: &Mat) -> Self {
        Self::new_with_flags(A, SvdOptions::default())
    }

    pub(super) fn new_with_flags(A: &Mat, options: SvdOptions) -> Self {
        if A.cols() <= A.rows() {
            Self::tall(A, options)
        } else {
            // A =U  S  V'
            // A'=V  S' U'
            let tmp = {
                let At = A.transpose();

                Self::tall(&At, options)
            };

            Self {
                U: tmp.V, //matd_transpose(tmp.V);
                S: tmp.S.transpose(),
                V: tmp.U, //matd_transpose(tmp.U);
            }
        }
    }

    // Computes an SVD for square or tall matrices. This code doesn't work
    // for wide matrices, because the bidiagonalization results in one
    // non-zero element too far to the right for us to rotate away.
    //
    // Caller is responsible for destroying U, S, and V.
    fn tall(A: &Mat, options: SvdOptions) -> Self {
        let mut B = A.clone();

        // Apply householder reflections on each side to reduce A to
        // bidiagonal form. Specifically:
        //
        // A = LS*B*RS'
        //
        // Where B is bidiagonal, and LS/RS are unitary.
        //
        // Why are we doing this? Some sort of transformation is necessary
        // to reduce the matrix's nz elements to a square region. QR could
        // work too. We need nzs confined to a square region so that the
        // subsequent iterative process, which is based on rotations, can
        // work. (To zero out a term at (i,j), our rotations will also
        // affect (j,i).
        //
        // We prefer bidiagonalization over QR because it gets us "closer"
        // to the SVD, which should mean fewer iterations.

        // LS: cumulative left-handed transformations
        let mut LS = Mat::identity(A.rows());

        // RS: cumulative right-handed transformations.
        let mut RS = Mat::identity(A.cols());

        for hhidx in 0..A.rows() {
            if hhidx < A.cols() {
                // We construct the normal of the reflection plane: let u
                // be the vector to reflect, x =[ M 0 0 0 ] the target
                // location for u (u') after reflection (with M = ||u||).
                //
                // The normal vector is then n = (u - x), but since we
                // could equally have the target location be x = [-M 0 0 0
                // ], we could use n = (u + x).
                //
                // We then normalize n. To ensure a reasonable magnitude,
                // we select the sign of M so as to maximize the magnitude
                // of the first element of (x +/- M). (Otherwise, we could
                // end up with a divide-by-zero if u[0] and M cancel.)
                //
                // The householder reflection matrix is then H=(I - nn'), and
                // u' = Hu.
                //
                //
                let vlen = A.rows() - hhidx;

                let mut v = calloc::<f64>(vlen);
                let mut mag2 = 0.;
                for i in 0..vlen {
                    v[i] = B[(hhidx+i, hhidx)];
                    mag2 += v[i]*v[i];
                }
                
                let oldv0 = v[0];
                if oldv0 < 0. {
                    v[0] -= mag2.sqrt();
                } else {
                    v[0] += mag2.sqrt();
                }

                mag2 += -oldv0*oldv0 + v[0]*v[0];

                // normalize v
                let mag = mag2.sqrt();

                // this case arises with matrices of all zeros, for example.
                if mag == 0. {
                    continue;
                }

                for i in 0..vlen {
                    v[i] /= mag;
                }

                // Q = I - 2vv'
                //matd_t *Q = matd_identity(A.rows());
                //for i in 0..vlen {
                //  for j in 0..vlen {
                //    Q[(i+hhidx, j+hhidx)] -= 2*v[i]*v[j];


                // LS = Mat::op("F*M", LS, Q);
                // Implementation: take each row of LS, compute dot product with n,
                // subtract n (scaled by dot product) from it.
                for i in 0..LS.rows() {
                    let mut dot = 0.;
                    for j in 0..vlen {
                        dot += LS[(i, hhidx + j)] * v[j];
                    }
                    for j in 0..vlen {
                        LS[(i, hhidx + j)] -= 2. * dot * v[j];
                    }
                }

                //  B = Mat::op("M*F", Q, B); // should be Q', but Q is symmetric.
                for i in 0..B.cols() {
                    let mut dot = 0.;
                    for j in 0..vlen {
                        dot += B[(hhidx + j, i)] * v[j];
                    }
                    for j in 0..vlen {
                        B[(hhidx + j, i)] -= 2. * dot * v[j];
                    }
                }
            }

            if hhidx+2 < A.cols() {
                let vlen = A.cols() - hhidx - 1;

                let mut v = calloc::<f64>(vlen);

                let mut mag2 = 0.;
                for i in 0..vlen {
                    v[i] = B[(hhidx, hhidx + i + 1)];
                    mag2 += v[i]*v[i];
                }

                let oldv0 = v[0];
                v[0] = if oldv0 < 0. {
                    -mag2.sqrt()
                } else {
                    mag2.sqrt()
                };

                mag2 += -oldv0*oldv0 + v[0]*v[0];

                // compute magnitude of ([1 0 0..]+v)
                let mag = mag2.sqrt();

                // this case can occur when the vectors are already perpendicular
                if mag == 0. {
                    continue;
                }

                for i in 0..vlen {
                    v[i] /= mag;
                }

                // TODO: optimize these multiplications
                // matd_t *Q = matd_identity(A.cols());
                //  for i in 0..vlen {
                //    for j in 0..vlen {
                //       Q[(i+1+hhidx, j+1+hhidx)] -= 2*v[i]*v[j];

                //  RS = Mat::op("F*M", RS, Q);
                for i in 0..RS.rows() {
                    let mut dot = 0.;
                    for j in 0..vlen {
                        dot += RS[(i, hhidx+1+j)] * v[j];
                    }
                    for j in 0..vlen {
                        RS[(i, hhidx+1+j)] -= 2.*dot*v[j];
                    }
                }

                //   B = Mat::op("F*M", B, Q); // should be Q', but Q is symmetric.
                for i in 0..B.rows() {
                    let mut dot = 0.;
                    for j in 0..vlen {
                        dot += B[(i, hhidx+1+j)] * v[j];
                    }
                    for j in 0..vlen {
                        B[(i, hhidx+1+j)] -= 2.*dot*v[j];
                    }
                }
            }
        }

        // maxiters used to be smaller to prevent us from looping forever,
        // but this doesn't seem to happen any more with our more stable
        // svd22 implementation.
        const MAX_ITERS: u32 = 1u32 << 30;
        assert!(MAX_ITERS > 0); // reassure clang

        // let mut maxv: f64; // maximum non-zero value being reduced this iteration

        // which method will we use to find the largest off-diagonal
        // element of B?
        if true {//(B.cols() < 6) ? 2 : 1;
            let method = FindMaxFast::new(&B);
            find_max(&options, &mut B, &mut LS, &mut RS, method);
        } else {
            find_max(&options, &mut B, &mut LS, &mut RS, FindMaxReference {});
        };

        fn find_max(options: &SvdOptions, B: &mut Mat, LS: &mut Mat, RS: &mut Mat, mut find_max_method: impl FindMax) {
            for iter in 0..MAX_ITERS {
                // No diagonalization required for 0x0 and 1x1 matrices.
                if B.cols() < 2 {
                    break;
                }
    
                // find the largest off-diagonal element of B, and put its
                // coordinates in maxi, maxj.
                let (maxi, maxj) = match find_max_method.iter(B) {
                    Some(idxs) => idxs,
                    None => break,
                };
    
        //        printf(">>> %5d %3d, %3d %15g\n", maxi, maxj, iter, maxv);
    
                // Now, solve the 2x2 SVD problem for the matrix
                // [ A0 A1 ]
                // [ A2 A3 ]
                let A0: f64 = B[(maxi, maxi)];
                let A1: f64 = B[(maxi, maxj)];
                let A2: f64 = B[(maxj, maxi)];
                let A3: f64 = B[(maxj, maxj)];
    
                if true {
                    let AQ = [ A0, A1, A2, A3 ];
    
                    let (U, _S, V) = svd22(AQ);
    
        /*  Reference (slow) implementation...
    
                    // LS = LS * ROT(theta) = LS * QL
                    matd_t *QL = matd_identity(A.rows());
                    QL[(maxi, maxi)] = U[0];
                    QL[(maxi, maxj)] = U[1];
                    QL[(maxj, maxi)] = U[2];
                    QL[(maxj, maxj)] = U[3];
    
                    matd_t *QR = matd_identity(A.cols());
                    QR[(maxi, maxi)] = V[0];
                    QR[(maxi, maxj)] = V[1];
                    QR[(maxj, maxi)] = V[2];
                    QR[(maxj, maxj)] = V[3];
    
                    LS = Mat::op("F*M", LS, QL);
                    RS = Mat::op("F*M", RS, QR); // remember we'll transpose RS.
                    B = Mat::op("M'*F*M", QL, B, QR);
    
                    matd_destroy(QL);
                    matd_destroy(QR);
        */
    
                    //  LS = Mat::op("F*M", LS, QL);
                    for i in 0..LS.rows() {
                        let vi = LS[(i, maxi)];
                        let vj = LS[(i, maxj)];
    
                        LS[(i, maxi)] = U[0]*vi + U[2]*vj;
                        LS[(i, maxj)] = U[1]*vi + U[3]*vj;
                    }
    
                    //  RS = Mat::op("F*M", RS, QR); // remember we'll transpose RS.
                    for i in 0..RS.rows() {
                        let vi = RS[(i, maxi)];
                        let vj = RS[(i, maxj)];
    
                        RS[(i, maxi)] = V[0]*vi + V[2]*vj;
                        RS[(i, maxj)] = V[1]*vi + V[3]*vj;
                    }
    
                    // B = Mat::op("M'*F*M", QL, B, QR);
                    // The QL matrix mixes rows of B.
                    for i in 0..B.cols() {
                        let vi = B[(maxi, i)];
                        let vj = B[(maxj, i)];
    
                        B[(maxi, i)] = U[0]*vi + U[2]*vj;
                        B[(maxj, i)] = U[1]*vi + U[3]*vj;
                    }
    
                    // The QR matrix mixes columns of B.
                    for i in 0..B.rows() {
                        let vi = B[(i, maxi)];
                        let vj = B[(i, maxj)];
    
                        B[(i, maxi)] = V[0]*vi + V[2]*vj;
                        B[(i, maxj)] = V[1]*vi + V[3]*vj;
                    }
                }
    
                if !options.suppress_warnings && iter == MAX_ITERS {
                    //TODO: fixme
                    println!("WARNING: maximum iters (maximum = {})", iter);
                    // println!("WARNING: maximum iters (maximum = {}, matrix {} x {}, max={:.15})", iter, A.rows(), A.cols(), maxv);
                    // matd_print(A, "%15f");
                }
            }
        }

        // them all positive by flipping the corresponding columns of
        // U/LS.
        let mut idxs = calloc::<usize>(A.cols());
        let mut vals = calloc::<f64>(A.cols());
        for i in 0..A.cols() {
            idxs[i] = i;
            vals[i] = B[(i, i)];
        }

        // A bubble sort. Seriously.
        //TODO: better sort
        loop {
            let mut changed = false;

            for i in 0..(A.cols() - 1) {
                if vals[i+1].abs() > vals[i].abs() {
                    idxs.swap(i, i+1);
                    vals.swap(i, i+1);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut LP = Mat::identity(A.rows());
        let mut RP = Mat::identity(A.cols());

        for i in 0..A.cols() {
            LP[(idxs[i], idxs[i])] = 0.; // undo the identity above
            RP[(idxs[i], idxs[i])] = 0.;

            LP[(idxs[i], i)] = if vals[i] < 0. { -1. } else { 1. };
            RP[(idxs[i], i)] = 1.; //vals[i] < 0 ? -1 : 1;
        }
        std::mem::drop(idxs);
        std::mem::drop(vals);

        // we've factored:
        // LP*(something)*RP'

        // solve for (something)
        B = Mat::op("M'*F*M", &[&LP, &B, &RP]).unwrap();

        // update LS and RS, remembering that RS will be transposed.
        let LS = Mat::op("F*M", &[&LS, &LP]).unwrap();
        std::mem::drop(LP);
        let RS = Mat::op("F*M", &[&RS, &RP]).unwrap();
        std::mem::drop(RP);

        // make B exactly diagonal
        for i in 0..B.rows() {
            for j in 0..B.cols() {
                if i != j {
                    B[(i, j)] = 0.;
                }
            }
        }

        MatSVD {
            U: LS,
            S: B,
            V: RS,
        }
    }
}