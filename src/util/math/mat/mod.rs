mod mat;
/// Choelsky Decomposition
mod chol;
/// PLU Decomposition
mod plu;
/// SVD decomposition
mod svd;
/// matd_op
mod op;
/// Matrix dimensions and indices
mod index;
mod mat22;
mod mat33;

pub(crate) use mat::Mat;
pub(crate) use mat22::Mat22;
pub(crate) use mat33::Mat33;

pub(crate) use index::{MatDims, MatIndex, OutOfBoundsError};
pub(crate) use chol::MatChol;
#[allow(unused_imports)]
pub(crate) use plu::MatPLU;
#[allow(unused_imports)]
pub(crate) use svd::{SvdOptions, MatSVD};