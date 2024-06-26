use std::{time::{Instant, Duration}, collections::HashMap, fmt::Display, borrow::Cow};

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
        let mut last_time = tp.start();
        for stamp in tp.stamps.iter() {
            let name = stamp.name();

            let duration = stamp.wall - last_time;
            last_time = stamp.wall;
            
            match self.values.entry(name.into()) {
                std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().push(duration),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(Vec::new()).push(duration);
                    self.keys.push(name.into());
                },
            }
        }
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

/// Records 
#[cfg_attr(feature="python", pyo3::pyclass(module="apriltag_rs"))]
#[derive(Clone, Debug)]
pub struct TimeProfile {
    /// Start timestamp
    now: Instant,
	now_proc: Option<cpu_time::ProcessTime>,
    /// Named timestamps
    stamps: Vec<TimeProfileEntry>,
}

impl Default for TimeProfile {
    fn default() -> Self {
        Self {
            now: Instant::now(),
			now_proc: cpu_time::ProcessTime::try_now().ok(),
            stamps: Default::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TimeProfileEntry {
    /// Entry name
    name: Cow<'static, str>,
    /// Entry timestamp
    wall: Instant,
	proc: Option<cpu_time::ProcessTime>,
}

impl TimeProfileEntry {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
    pub(crate) fn timestamp(&self) -> &Instant {
        &self.wall
    }
}

impl TimeProfile {
    /// Get start time
    pub(crate) fn start(&self) -> Instant {
        self.now
    }

    pub(crate) fn set_start(&mut self, value: Instant) {
        self.now = value;
        #[cfg(debug_assertions)]
        if let Some(first_entry) = self.stamps.first() {
            assert!(first_entry.wall >= value);
        }
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.stamps.clear();
        self.now = Instant::now();
    }

    /// Record a timestamp right now
    #[inline]
    pub fn stamp(&mut self, name: impl Into<Cow<'static, str>>) {
        let name = name.into();
        let timestamp = Instant::now();
		let proc = cpu_time::ProcessTime::try_now().ok();
        self.stamp_at_inner(name, timestamp, proc)
    }

    /// Mark a specific time
    pub fn stamp_at(&mut self, name: impl Into<Cow<'static, str>>, timestamp: Instant) {
        let name = name.into();
        self.stamp_at_inner(name, timestamp, None);
    }

    #[inline]
    fn stamp_at_inner(&mut self, name: Cow<'static, str>, timestamp: Instant, proc: Option<cpu_time::ProcessTime>) {
        let entry = TimeProfileEntry {
            name,
            wall: timestamp,
			proc,
        };

        self.stamps.push(entry);
    }

    /// Get duration from [start] to last recorded timestamp
    pub fn total_duration(&self) -> Duration {
        let stamps = &self.stamps;
        if stamps.len() == 0 {
            return Duration::ZERO;
        }

        let first = stamps.first().unwrap();
        let last = stamps.last().unwrap();

        last.wall - first.wall
    }

    pub(crate) fn entries(&self) -> &[TimeProfileEntry] {
        &self.stamps
    }
}

impl Display for TimeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stamps = &self.stamps;

        // Find maximums for scaling output
        let max_name_length = stamps.iter()
            .map(|stamp| stamp.name.len())
            .max()
            .unwrap_or(0)
            .max(1);

        let total_time = match stamps.last() {
            Some(last_stamp) => last_stamp.wall - self.now,
            None => Duration::ZERO,
        };

		let total_proc =  stamps.iter().rev()
			.find_map(|stamp| stamp.proc)
			;

		if f.alternate() {
			write!(f, "{:>2} {:width$} {:>12} ms {:>12} ms {:>4}", "#", "Name", "Wall", "Wall (cum)", "%", width=max_name_length)?;
			if total_proc.is_some() {
				write!(f, " {:>12} ms {:>4}", "Proc", "P%")?;
			}
			writeln!(f)?;
		}
        let mut last_time = self.now;
		let mut last_proc = self.now_proc;
        for (i, stamp) in stamps.iter().enumerate() {
            let cumtime = stamp.wall - self.now;

            let parttime = stamp.wall - last_time;

            write!(f, "{:2} {:0width$} {:12.6} ms {:12.6} ms {:3.0}%",
                i,
                stamp.name,
                parttime.as_secs_f64() * 1000.,
                cumtime.as_secs_f64() * 1000.,
                100. * parttime.as_secs_f64() / total_time.as_secs_f64(),
                width=max_name_length
            )?;
			
			if let Some(proc) = stamp.proc {
				if let Some(last_proc_) = last_proc {
					let proc_part = proc.duration_since(last_proc_);
					write!(f, " {:12.6} ms",
						proc_part.as_secs_f64() * 1000.,
					)?;
				}
				last_proc = stamp.proc;
			}

			writeln!(f)?;

            last_time = stamp.wall;
        }
        Ok(())
    }
}