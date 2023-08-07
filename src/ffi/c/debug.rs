use crate::TimeProfile;

use super::shim::{InPtr, cffi_wrapper, ReadPtr};

pub type timeprofile_t = TimeProfile;

#[no_mangle]
pub unsafe extern "C" fn timeprofile_display(tp: *const timeprofile_t) {
    if let Some(tp) = tp.as_ref() {
        print!("{tp}");
    }
}

#[no_mangle]
pub unsafe extern "C" fn timeprofile_total_utime<'a>(tp: InPtr<'a, timeprofile_t>) -> u64 {
    cffi_wrapper(|| {
        let tp = tp.try_ref("tp")?;
        let total = tp.total_duration();
        let utime = total.as_micros();
        if utime > u64::MAX as u128 {
            Ok(u64::MAX)
        } else {
            Ok(utime as u64)
        }
    })
}