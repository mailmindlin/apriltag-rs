use crate::TimeProfile;


pub type timeprofile_t = TimeProfile;
#[no_mangle]
pub unsafe extern "C" fn timeprofile_display(tp: *const timeprofile_t) {
    if let Some(tp) = tp.as_ref() {
        print!("{tp}");
    }
}

#[no_mangle]
pub unsafe extern "C" fn timeprofile_total_utime(tp: *const timeprofile_t) -> u64 {
    if let Some(tp) = tp.as_ref() {
        let total = tp.total_duration();
        let utime = total.as_micros();
        if utime > u64::MAX as u128 {
            u64::MAX
        } else {
            utime as u64
        }
    } else {
        0
    }
}