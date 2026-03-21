//! Debug helpers
mod timeprofile;

pub use timeprofile::{TimeProfile, TimeProfileStatistics};
pub(crate) use timeprofile::TimeProfileEntry;

/// File names for debug images
#[cfg(feature="debug")]
#[allow(unused)]
pub(crate) mod debug_images {
    /// Raw source image
    pub(crate) const SOURCE: &str = "00_debug_src.pnm";
    /// Image after quad decimation
    pub(crate) const DECIMATE: &str = "00a_debug_decimate.pnm";

    /// Image after preprocessing
    pub(crate) const PREPROCESS: &str = "01_debug_preprocess.pnm";

    pub(crate) const THRESHOLD: &str = "02_debug_threshold.pnm";
    pub(crate) const TILE_MIN: &str = "02a_tile_minmax_min.pnm";
    pub(crate) const TILE_MAX: &str = "02b_tile_minmax_max.pnm";
    pub(crate) const BLUR_MIN: &str = "02c_tile_minmax_blur_min.pnm";
    pub(crate) const BLUR_MAX: &str = "02d_tile_minmax_blur_max.pnm";

    pub(crate) const SEGMENTATION: &str = "03a_debug_segmentation.pnm";
    pub(crate) const UNIONFIND_DEPTH: &str = "03b_debug_uniofind_depth.pnm";
    pub(crate) const CLUSTERS: &str = "04_debug_clusters.pnm";
    pub(crate) const LINES: &str = "05_debug_lines.ps";
    pub(crate) const QUADS: &str = "07_debug_quads.pnm";
    pub(crate) const SAMPLES: &str = "08_debug_samples.pnm";
    pub(crate) const QUADS_FIXED: &str = "09a_debug_quads_fixed.pnm";
    pub(crate) const QUADS_PS: &str = "09b_debug_quads.ps";

    pub(crate) const OUTPUT: &str = "10a_debug_output.pnm";
    pub(crate) const OUTPUT_PS: &str = "10b_debug_output.ps";
}