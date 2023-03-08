use std::{path::Path, io::{self, BufReader, BufRead, Read}, fs::OpenOptions};

use crate::util::mem::calloc;

#[derive(PartialEq, Eq)]
pub(super) enum PNMFormat {
    Binary = 4,
    Gray = 5,
    RGB = 6,
}

pub(super) struct PNM {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) format: PNMFormat,
    /// 1 = binary, 255 = one byte, 65535 = two bytes
    pub(super) max: usize,

    // if max=65535, in big endian
    pub(super) buf: Box<[u8]>,
}

impl PNM {
    pub(super) fn create_from_file(path: &Path) -> io::Result<Self> {
        let f = OpenOptions::new()
            .read(true)
            .open(path)?;
        
        let mut br = BufReader::new(f);

        // will be 3 when we're all done.
        let mut format = None;
        let mut params = Vec::new();

        while params.len() < 3 && !(format == Some(PNMFormat::Binary) && params.len() == 2) {
            let mut line = String::new();
            br.read_line(&mut line)?;
            let mut tmp = line.chars().peekable();

            match tmp.peek() {
                // skip comments
                Some('#') => continue,
                Some('P') if format == None => {
                    tmp.next().unwrap();
                    format = match tmp.next().unwrap() {
                        '4' => Some(PNMFormat::Binary),
                        '5' => Some(PNMFormat::Gray),
                        '6' => Some(PNMFormat::RGB),
                        c => panic!("Unsupported format id: {}", c),
                    };
                }
                _ => {},
            }
            

            // pull integers out of this line until there are no more.
            while params.len() < 3 && tmp.peek().is_some() {
                while tmp.next_if_eq(&' ').is_some() {
                }

                // encounter rubbish? (End of line?)
                let mut found_any = false;
                let mut acc = 0;
                'digits: while let Some(p) = tmp.peek() {
                    let d = match p {
                        '0' => 0,
                        '1' => 1,
                        '2' => 2,
                        '3' => 3,
                        '4' => 4,
                        '5' => 5,
                        '6' => 6,
                        '7' => 7,
                        '8' => 8,
                        '9' => 9,
                        _ => break 'digits
                    };
                    tmp.next().unwrap();

                    acc = (acc * 10) + d;
                    found_any = true;
                };

                if !found_any {
                    break;
                }
                params.push(acc);

                tmp.next().unwrap();
            }
        }

        let format = format.unwrap();

        let width = params[0];
        let height = params[1];
        let mut max = *params.get(2).unwrap_or(&0);

        let buflen = match format {
            PNMFormat::Binary => {
                // files in the wild sometimes simply don't set max
                max = 1;

                height * ((width + 7) / 8)
            },
            PNMFormat::Gray =>
                match max {
                    255 => width * height,
                    65535 => 2 * width * height,
                    _ => panic!(),
                },
            PNMFormat::RGB =>
                match max {
                    255 => width * height * 3,
                    65535 => 2 * width * height * 3,
                    _ => panic!(),
                },
        };
        let mut buf = calloc::<u8>(buflen);
        br.read_exact(&mut buf)?;
        
        Ok(Self {
            width,
            height,
            format,
            max,
            buf,
        })
    }
}