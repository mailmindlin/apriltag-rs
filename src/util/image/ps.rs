use std::io::{self, Write};


pub struct PostScriptWriter<W: io::Write> {
    inner: W,
}

impl<W: Write> Write for PostScriptWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: io::Write> PostScriptWriter<W> {
    
}