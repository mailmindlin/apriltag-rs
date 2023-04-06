use std::time::{Instant, Duration};

pub struct TimeProfile {
    now: Instant,

    stamps: Vec<TimeProfileEntry>,
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

        let stamps = &self.stamps;
        let max_name = stamps.iter()
            .map(|stamp| stamp.name.len())
            .max()
            .unwrap_or(0);

        for stamp in stamps.iter() {
            let cumtime = stamp.utime - self.now;

            let parttime = stamp.utime - last_time;

            println!("{:2} {:0width$} {:12.3} ms {:12.3} ms", i, stamp.name, parttime.as_micros() as f64 / 1000.0, cumtime.as_micros() as f64 / 1000.0, width=max_name);

            i += 1;
            last_time = stamp.utime;
        }
    }

    pub fn total_utime(&self) -> Duration {
        let stamps = &self.stamps;
        if stamps.len() == 0 {
            return Duration::ZERO;
        }

        let first = stamps.first().unwrap();
        let last = stamps.last().unwrap();

        last.utime - first.utime
    }
}