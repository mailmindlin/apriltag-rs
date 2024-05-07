#![allow(non_camel_case_types, non_snake_case)]
mod writelist;

use std::{time::{Instant, Duration}, collections::HashMap, sync::{Mutex, Condvar, Arc}, thread::{available_parallelism, self, sleep, ScopedJoinHandle}, cmp, num::{NonZeroUsize, NonZeroU32}};

use apriltag_rs::families::{rotations, generator::{hamming_distance, ImageLayout}};
use apriltag_rs::families::generator::TagFamily;
use clap::Parser;
use rand::{Rng, SeedableRng, rngs::StdRng};
use writelist::{WriteList, ListSnapshot};


#[derive(Parser)]
struct CLIArgs {
    layout: String,
    min_hamming: u32,
}

const PRIME: u64 = 982451653;
/*
static final void printBoolean(PrintStream outs, long v, int nbits)
{
    for (int b = nbits-1; b >= 0; b--)
        outs.printf("%d", (v & (1L<<b)) > 0 ? 1 : 0);
}

static final void printCodes(long codes[], int nbits)
{
    for (int i = 0; i < codes.length; i++) {
        long w = codes[i];
        System.out.printf("%5d ", i);
        printBoolean(System.out, w, nbits);
        println!("    %0"+((int) Math.ceil(nbits/4))+"x", w);
    }
}
*/

fn isCodePartiallyOkay(config: &Config, v: u64, rotcodes: ListSnapshot<u64>, nRotCodesPartial: usize) -> bool {
    // tag must be reasonably complex
    if !isComplexEnough(&config.layout, v) {
        return false;
    }

    // The tag must be different from itself when rotated.
    let rv = rotations(v, config.nbits() as _);

    for i in 0..(rv.len() - 1) {
        for j in (i+1)..rv.len() {
            if !hammingDistanceAtLeast(rv[i], rv[j], config.min_hamming) {
                return false;
            }
        }
    }

    // tag must be different from other tags.
    for w in rotcodes.iter().take(nRotCodesPartial as usize) {
        if !hamming_distance(v, *w) >= config.min_hamming {
            return false;
        }
    }

    true
}

fn isComplexEnough(layout: &ImageLayout, v: u64) -> bool {
    let tag = layout.render_to_array(v);
    let mut energy = 0;
    for y in 0..tag.len() {
        for x in 0..(tag.len() - 1) {
            if (tag[y][x] == 0 && tag[y][x + 1] == 1) || (tag[y][x] == 1 && tag[y][x + 1] == 0) {
                energy += 1;
            }
        }
    }

    for x in 0..tag.len() {
        for y in 0..(tag.len() - 1) {
            if (tag[y][x] == 0 && tag[y+1][x] == 1) || (tag[y][x] == 1 && tag[y+1][x] == 0) {
                energy += 1;
            }
        }
    }

    let mut area = 0;
    for y in 0..tag.len() {
        for x in 0..tag.len() {
            if tag[y][x] == 0 || tag[y][x] == 1 {
                area += 1;
            }
        }
    }

    let max_energy = 2 * area;
    energy as f64 >= 0.3333 * max_energy as f64
}

fn union(components: &mut Vec<Vec<i32>>, x0: usize, y0: usize, x1: usize, y1: usize) {
    let c0 = components[y0][x0];
    let c1 = components[y1][x1];

    for y in 0..components.len() {
        for x in 0..components.len() {
            if components[y][x] == c1 {
                components[y][x] = c0;
            }
        }
    }
}

fn isCodePartiallyOkay2(v: u64, rotcodes: ListSnapshot<u64>, nRotCodesPartial: usize, min_hamming: u32) -> bool {
    // tag must be different from other tags.
    for w in rotcodes.iter().skip(nRotCodesPartial) {
        if !hammingDistanceAtLeast(v, *w, min_hamming) {
            return false;
        }
    }
    true
}

struct PartialApprovalTask<'a> {
    config: Arc<Config>,
    map: &'a Mutex<HashMap<u64, PartialApprovalResult>>,
    rotcodes: &'a WriteList<u64>,
    nRotCodesPartial: usize,
    V0: u64,
    iter0: u64,
    iter1: u64,
}

impl<'a> PartialApprovalTask<'a> {
    fn run(&mut self) {
        // compute v = V0 + PRIME * iter0,
        // being very careful about overflow.
        // (consider the power-of-two expansion of iter0....)
        let mut v = self.V0;

        let mut acc = PRIME;
        let mut M = self.iter0;
        let mask: u64 = (1 << self.config.nbits()) - 1;
        while M > 0 {
            if (M & 1) > 0 {
                v += acc;
                v &= mask;
            }

            acc *= 2;
            acc &= mask;
            M >>= 1;
        }

        let mut good_codes = Vec::new();
        for iter in self.iter0..self.iter1 {
            v += PRIME; // big prime.
            v &= mask;

            if isCodePartiallyOkay(&self.config, v, self.rotcodes.snapshot(), self.nRotCodesPartial) {
                good_codes.push(v);
            }
        }

        let result = PartialApprovalResult {
            good_codes,
            nRotCodesPartial: self.nRotCodesPartial,
            iter0: self.iter0,
            iter1: self.iter1,
        };

        {
            let mut map = self.map.lock().unwrap();
            map.insert(result.iter0, result);
        }
    }
}

struct PartialApprovalResult {
    good_codes: Vec<u64>,
    nRotCodesPartial: usize,
    iter0: u64,
    iter1: u64,
}

struct Signal {
    mux: Mutex<bool>,
    cond: Condvar,
}

impl Signal {
    fn new() -> Self {
        Self {
            mux: Mutex::new(false),
            cond: Condvar::new(),
        }
    }
    fn wait(&self, duration: Duration) -> bool {
        let guard = self.mux.lock().unwrap();
        let (guard, res) = self.cond.wait_timeout(guard, duration).unwrap();
        *guard
    }

    fn signal(&self) {
        let mut lock = self.mux.lock().unwrap();
        *lock = true;
        self.cond.notify_all();
    }
}

fn reporting_thread(codelist: &WriteList<u64>, stop: &Signal) {
    let mut lastReportTime = Instant::now();
    let mut lastNumCodes = 0;

    loop {
        // print a partial report.
        if lastReportTime.elapsed() > Duration::from_secs(60 * 60) || (codelist.len() as f64 > (1.1 * lastNumCodes as f64) && lastReportTime.elapsed() > Duration::from_secs(60)) {
            report();
            lastReportTime = Instant::now();
            lastNumCodes = codelist.len();
        }

        if stop.wait(Duration::from_secs(10)) {
            break;
        }
    }
}

fn approval_thread(map: &Mutex<HashMap<u64, PartialApprovalResult>>, rotcodes: &WriteList<u64>, codelist: &WriteList<u64>, config: Arc<Config>) {
    fn get_result(map: &Mutex<HashMap<u64, PartialApprovalResult>>, iter: &mut u64) -> PartialApprovalResult {
        loop {
            {
                let map = map.get_mut().unwrap();
                match map.remove(&iter) {
                    Some(result) => {
                        *iter = result.iter1;
                        return result;
                    },
                    None => {
                        sleep(Duration::from_millis(1));
                    }
                }
            }
        }
    }

    let mut iter = 0;
    loop {
        if iter == (1 << config.nbits()) {
            return;
        }

        let result = get_result(map, &mut iter);

        for v in result.good_codes {
            if !isCodePartiallyOkay2(v, rotcodes.snapshot(), result.nRotCodesPartial, config.min_hamming) {
                continue;
            }

            codelist.push(v);

            let rv = rotations(v, config.nbits() as u64);

            rotcodes.extend(rv);
        }
    }
}

fn compute(config: Arc<Config>) -> TagFamily {
    let map = Mutex::new(HashMap::new());
    let mut codelist = WriteList::new(); // code lists
    let mut rotcodes = WriteList::new();
    let starttime = Instant::now();

    // begin our search at a random position to avoid any bias
    // towards small numbers (which tend to have larger regions of
    // solid black).
    let mut rng = StdRng::seed_from_u64(config.nbits() as u64 * 10_000 + config.min_hamming as u64 * 100 + 7);
    let V0: u64 = rng.gen();

    let mut lastprogresstime = starttime;
    let mut lastprogressiters = 0;

    let nthreads = available_parallelism().unwrap_or(NonZeroUsize::new(16).unwrap()).get();
    println!("Using {nthreads} threads.");

    thread::scope(|s| {
        let approvalThread = {
            thread::Builder::new()
                .name("ApprovalThread".into())
                .spawn_scoped(s, || approval_thread(&map, &rotcodes, &codelist, config.clone()))
        }.unwrap();

        let reporting_stop = Signal::new();

        let reportingThread = {
            thread::Builder::new()
                .name("ReportingThread".into())
                .spawn_scoped(s, || reporting_thread(&codelist, &reporting_stop))
                .unwrap()
        };

        let mut iter = 0;
        const MAP_MAX: usize = 300;
        const CHUNKSIZE: u64 = 50_000;

        let mut queue = Vec::<ScopedJoinHandle<()>>::new();

        loop {
            // print a progress report.
            let now = Instant::now();
            if now - lastprogresstime > Duration::from_secs(5) {

                let donepercent = (iter as f64 *100.0)/((1 << config.nbits()) as f64);
                let dt = (now - lastprogresstime).as_secs_f64();
                let diters = iter - lastprogressiters;
                let rate = diters as f64 / dt; // iterations per second
                let secremaining = ((1u64 << config.nbits()) - iter) as f64 / rate;
                println!("{donepercent:8.4}%  codes: {:-5} ({rate:.0} iters/sec, {:.2} minutes = {:.2} hours)           ", codelist.len(), secremaining / 60., secremaining/3600.);
                lastprogresstime = now;
                lastprogressiters = iter;
            }

            queue = queue.into_iter()
                .filter(|handle| !handle.is_finished())
                .collect();

            if queue.len() < nthreads {
                if iter < 1 << config.layout.num_bits() {
                    let add_task = {
                        let map = map.lock().unwrap();
                        map.len() < MAP_MAX
                    };

                    if add_task {
                        {
                            let iter0 = iter;
                            iter = cmp::min(iter + CHUNKSIZE, 1 << config.nbits());

                            let task = PartialApprovalTask {
                                config,
                                map: &map,
                                rotcodes: &rotcodes,
                                nRotCodesPartial: codelist.len(),
                                V0,
                                iter0,
                                iter1: iter,
                            };
                            let handle = s.spawn(move || task.run());
                            queue.push(handle);
                        }

                        sleep(Duration::from_millis(1));
                        continue;
                    }
                }
            }

            if approvalThread.is_finished() {
                println!("Approval thread dead. Done!");
                break;
            }

            sleep(Duration::from_millis(10));
        }

        reporting_stop.signal();

        let codes = codelist.snapshot()
            .iter()
            .copied()
            .collect();

        TagFamily {
            min_hamming_distance: NonZeroU32::new(config.min_hamming).unwrap(),
            codes,
            layout: config.layout,
        }
    })
}

fn report(starttime: Instant, codelist: &WriteList<u64>, config: &Config) {
    let codes = codelist.snapshot();

    let mut hds = vec![0; config.nbits() + 1];
    let mut hd_total = 0;

    // compute hamming distance table
    for (i, code) in codes.iter().enumerate() {
        let rvs = rotations(*code, config.nbits() as _);

        for (j, code2) in codes.iter().enumerate().skip(i+1) {
            let dist = rvs.iter()
                .map(|rv| hamming_distance(*rv, *code2))
                .min()
                .unwrap();

            hds[dist as usize] += 1;
            if dist < config.min_hamming {
                eprintln!("ERROR, dist = {dist:3}: {i} {j}");
            }
            hd_total += 1;
        }
    }

    println!("\n\npackage april.tag;\n\n");
    let cname = format!("Tag{}{}h{}", config.layout.name(), config.nbits(), config.min_hamming);
    println!("/** Tag family with {} distinct codes.", codes.len());
    println!("    bits: {},  minimum hamming: {}\n", config.nbits(), config.min_hamming);

    // compute some ROC statistics, assuming randomly-visible targets
    // as a function of how many bits we're willing to correct.
    println!("    Max bits corrected       False positive rate");

    for cbits in 0..=((config.min_hamming - 1) / 2) {
        let mut validCodes = 0; // how many input codes will be mapped to a single valid code?
        // it's the number of input codes that have 0 errors, 1 error, 2 errors, ..., cbits errors.
        for i in 0..=cbits {
            validCodes += choose(config.nbits() as _, i as _);
        }

        validCodes *= codes.len() as u64; // total number of codes

        println!("          {:3}             {:15.8} %", cbits, (100.0 * validCodes as f64) / ((1 << config.nbits()) as f64));
    }

    println!("\n    Generation time: {} s\n", starttime.elapsed().as_secs_f64());

    println!("    Hamming distance between pairs of codes (accounting for rotation):\n");
    for (i, hd) in hds.iter().enumerate() {
        println!("    {i:4}  {hd}");
    }

    println!("**/");

    println!("public class {cname} extends TagFamily {{");

    let maxLength = 8192;
    let numSubMethods = (maxLength + codes.len() - 1) / maxLength;
    for i in 0..numSubMethods {
        println!("\tprivate static class ConstructCodes{i} {{");
        println!("\t\tprivate static long[] constructCodes() {{");
        print!("\t\t\treturn new long[] {{ ");
        let jMax = cmp::min(maxLength, codes.len() - i * maxLength);
        for j in 0..jMax {
            let w = codes[i * maxLength + j];
            print!("0x{:0width$x}L", w, width=f32::ceil(config.nbits() as f32 / 4.) as usize);
            // print!("0x%0" + (Math.ceil(config.nbits() / 4) as i32) + "xL", w);
            if j == jMax - 1 {
                println!(" }};\n\t\t}}\n\t}}\n");
            } else {
                print!(", ");
            }
        }
    }

    println!("\tprivate static long[] constructCodes() {{");
    println!("\t\tlong[] codes = new long[{}];", codes.len());
    for i in 0..numSubMethods {
        println!("\t\tSystem.arraycopy(ConstructCodes{i}.constructCodes(), 0, codes, {}, {});", i * maxLength, cmp::min(maxLength, codes.len() - i * maxLength));
    }
    println!("\t\treturn codes;");
    println!("\t}}\n");


    println!("\tpublic {cname}() {{");
    println!("\t\tsuper(ImageLayout.Factory.createFromString(\"{}\", \"{}\"), {}, constructCodes());",
            config.layout.name(), /*layout.getDataString()*/ "[todo]", config.min_hamming);
    println!("\t}}");
    println!("}}");
    println!();
    println!();
}

fn choose(n: u64, c: u64) -> u64 {
    let mut v = 1;
    for i in 0..c {
        v *= n - i;
    }
    for i in 1..=c {
        v /= i;
    }
    v
}

fn hammingDistanceAtLeast(a: u64, b: u64, minval: u32) -> bool {
    hamming_distance(a, b) >= minval
}

struct Config {
    layout: ImageLayout,
    min_hamming: u32,
}

impl Config {
    const fn nbits(&self) -> usize {
        self.layout.num_bits()
    }
}

fn main() {
    let args = CLIArgs::parse();

    let layout = ImageLayout::for_name(args.layout)
        .unwrap();

    let config = Arc::new(Config {
        layout,
        min_hamming: args.min_hamming,
    });

    compute(config);
}