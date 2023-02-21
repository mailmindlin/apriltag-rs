use std::io::{self, Write};


pub struct PostScriptWriter<'a, W: io::Write> {
    inner: &'a mut W,
}

impl<'a, W: Write> Write for PostScriptWriter<'a, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<'a, W: io::Write> PostScriptWriter<'a, W> {
    pub fn new(inner: &'a mut W) -> Self {
        Self { inner }
    }
}