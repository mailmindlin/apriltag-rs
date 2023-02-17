use std::time::{Instant, Duration};

pub(crate) struct TimeProfile {
    pub now: Instant,

    pub stamps: Vec<TimeProfileEntry>,
}

impl Default for TimeProfile {
    fn default() -> Self {
        Self {
            now: Instant::now(),
            stamps: Default::default(),
        }
    }
}

struct TimeProfileEntry {
    name: String,
    utime: Instant,
}

impl TimeProfile {
    pub fn clear(&mut self) {
        self.stamps.clear();
        self.now = Instant::now();
    }

    #[inline]
    pub fn stamp(&mut self, name: &str) {
        let entry = TimeProfileEntry {
            name: String::from(name),
            utime: Instant::now(),
        };

        self.stamps.push(entry);
    }

    pub fn display(&self) {
        let mut last_time = self.now;

        let mut i = 0;

        for stamp in self.stamps {
            let cumtime = stamp.utime - self.now;

            let parttime = stamp.utime - last_time;

            println!("{:2} {} {:15} ms {:15} ms\n", i, stamp.name, parttime.as_micros() as f64 / 1000.0, cumtime.as_micros() as f64 / 1000.0);

            i += 1;
            last_time = stamp.utime;
        }
    }

    pub fn total_utime(&self) -> Duration {
        if self.stamps.len() == 0 {
            return Duration::ZERO;
        }

        let first = self.stamps.first().unwrap();
        let last = self.stamps.last().unwrap();

        last.utime - first.utime
    }
}