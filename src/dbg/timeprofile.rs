use std::{time::{Instant, Duration}, collections::HashMap, fmt::Display};

/// Tracks statistics of multiple time profiles
#[derive(Default)]
pub struct TimeProfileStatistics {
    /// Entry information
    values: HashMap<String, Vec<Duration>>,
    /// Entry keys, in order
    keys: Vec<String>,
}

impl TimeProfileStatistics {
    pub fn add(&mut self, tp: &TimeProfile) {
        let mut last_time = tp.now;
        for stamp in tp.stamps.iter() {
            let duration = stamp.timestamp - last_time;
            last_time = stamp.timestamp;
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
        print!("{self}");
    }
}

impl Display for TimeProfileStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_name = self.keys
            .iter()
            .map(|stamp| stamp.len())
            .max()
            .unwrap_or(0);

        writeln!(f, " # {:width$} {:>15} {:>15} {:>15} {:>15}", "Name", "Average", "Min", "Max", "Std.dev", width=max_name)?;

        for (i, key) in self.keys.iter().enumerate() {
            let entry = self.values.get(key).unwrap();
            let mut max = Duration::ZERO;
            let mut min = Duration::from_nanos(u64::MAX);
            let mut sum = 0.;
            let mut stddev_acc = 0.;
            for d in entry.iter().copied() {
                let d_s = d.as_secs_f64();
                sum += d_s;
                stddev_acc += d_s * d_s;
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

            writeln!(f, "{:2} {:0width$} {:12.6} ms {:12.6} ms {:12.6} ms {:12.6} ms", i, key, avg * 1e3, min.as_secs_f64() * 1e3, max.as_secs_f64() * 1e3, stddev * 1e3, width=max_name)?;
        }

        Ok(())
    }
}
#[derive(Clone)]
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

#[derive(Clone)]
struct TimeProfileEntry {
    /// Entry name
    name: String,
    /// Entry timestamp
    timestamp: Instant,
}

impl TimeProfile {
    pub fn clear(&mut self) {
        self.stamps.clear();
        self.now = Instant::now();
    }

    /// Record a timestamp
    #[inline]
    pub fn stamp(&mut self, name: &str) {
        let entry = TimeProfileEntry {
            name: String::from(name),
            timestamp: Instant::now(),
        };

        self.stamps.push(entry);
    }

    /// Get total 
    pub fn total_utime(&self) -> Duration {
        let stamps = &self.stamps;
        if stamps.len() == 0 {
            return Duration::ZERO;
        }

        let first = stamps.first().unwrap();
        let last = stamps.last().unwrap();

        last.timestamp - first.timestamp
    }
}

impl Display for TimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut last_time = self.now;

        let mut i = 0;

        let stamps = &self.stamps;
        let max_name_length = stamps.iter()
            .map(|stamp| stamp.name.len())
            .max()
            .unwrap_or(0);

        let total_time = stamps.last()
            .map(|stamp| stamp.timestamp - self.now)
            .unwrap_or(Duration::ZERO);

        for stamp in stamps.iter() {
            let cumtime = stamp.timestamp - self.now;

            let parttime = stamp.timestamp - last_time;

            writeln!(f, "{:2} {:0width$} {:12.6} ms {:12.6} ms {:3.0}%", i, stamp.name, parttime.as_secs_f64() * 1000., cumtime.as_secs_f64() * 1000., 100. * parttime.as_secs_f64() / total_time.as_secs_f64(), width=max_name_length)?;

            i += 1;
            last_time = stamp.timestamp;
        }
        Ok(())
    }
}