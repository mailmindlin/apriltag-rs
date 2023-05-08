use std::{time::{Instant, Duration}, collections::HashMap};

#[derive(Default)]
pub struct TimeProfileAcc {
    values: HashMap<String, Vec<Duration>>,
    keys: Vec<String>,
}

impl TimeProfileAcc {
    pub fn add(&mut self, tp: &TimeProfile) {
        let mut last_time = tp.now;
        for stamp in tp.stamps.iter() {
            let duration = stamp.utime - last_time;
            last_time = stamp.utime;
            match self.values.entry(stamp.name.clone()) {
                std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().push(duration),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(Vec::new()).push(duration);
                    self.keys.push(stamp.name.clone());
                },
            }
        }
    }

    pub fn display(&self) {
        let max_name = self.keys
            .iter()
            .map(|stamp| stamp.len())
            .max()
            .unwrap_or(0);

            println!(" # {:width$} {:>15} {:>15} {:>15} {:>15}", "Name", "Average", "Min", "Max", "Std.dev", width=max_name);

        for (i, key) in self.keys.iter().enumerate() {
            let entry = self.values.get(key).unwrap();
            let mut max = Duration::ZERO;
            let mut min = Duration::from_nanos(u64::MAX);
            let mut sum = 0.;
            let mut stddev_acc = 0.;
            for d in entry.iter().copied() {
                let d_s = d.as_secs_f64();
                sum += d_s;
                stddev_acc += (d_s * d_s);
                if d > max {
                    max = d;
                }
                if d < min {
                    min = d;
                }
            }
            let len = entry.len().max(1);
            let avg = sum / (len as f64);
            let stddev = (stddev_acc / (len as f64)).sqrt();

            println!("{:2} {:0width$} {:12.6} ms {:12.6} ms {:12.6} ms {:12.6} ms", i, key, avg * 1e3, min.as_secs_f64() * 1e3, max.as_secs_f64() * 1e3, stddev * 1e3, width=max_name);
        }
    }
}

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

            println!("{:2} {:0width$} {:12.6} ms {:12.6} ms", i, stamp.name, parttime.as_micros() as f64 / 1000.0, cumtime.as_micros() as f64 / 1000.0, width=max_name);

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