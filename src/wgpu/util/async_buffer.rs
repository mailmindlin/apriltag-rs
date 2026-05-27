use std::{task::Poll, pin::Pin, ops::Deref, marker::PhantomData, borrow::Borrow, mem::size_of};
use bytemuck::{Pod, AnyBitPattern};
use futures::{channel::oneshot::Receiver, Future};
use wgpu::{SubmissionIndex, BufferAsyncError, Device as GpuDevice, Buffer as GpuBuffer, BufferSlice as GpuBufferSlice, BufferView};

use crate::util::mem::{SafeZero, calloc};

#[must_use]
pub(in super::super) struct AsyncBufferView<'a, 'b> {
    buffer_slice: GpuBufferSlice<'a>,
    device: &'b GpuDevice,
    receiver: Receiver<Result<(), BufferAsyncError>>,
    sub_idx: Option<SubmissionIndex>,
}

impl<'a, 'b> AsyncBufferView<'a, 'b> {
    fn new(buffer_slice: GpuBufferSlice<'a>, device: &'b GpuDevice, sub_idx: Option<SubmissionIndex>) -> Self {
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).expect("Couldn't notify mapping")
        });

        AsyncBufferView {
            buffer_slice,
            device,
            receiver,
            sub_idx,
        }
    }
}

impl<'a, 'b> Future for AsyncBufferView<'a, 'b> {
    type Output = Result<(), BufferAsyncError>;

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Self::Output> {
        self.device.poll(match self.sub_idx.take() {
            Some(sub_idx) => wgpu::PollType::Wait { submission_index: Some(sub_idx), timeout: None },
            None => wgpu::PollType::Poll,
        });

        match self.receiver.try_recv() {
            Ok(Some(v)) => Poll::Ready(v),
            Ok(None) | Err(_) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

pub(super) struct MappedBufferGuard<B: Borrow<GpuBuffer>, E: AnyBitPattern> {
    buffer: B,
    view: BufferView,
    element: PhantomData<E>,
}

impl<B: Borrow<GpuBuffer>, E: AnyBitPattern> MappedBufferGuard<B, E> {
    fn new(buffer: B, view: BufferView) -> Self {
        Self {
            buffer,
            view,
            element: PhantomData,
        }
    }
}
impl<B: Borrow<GpuBuffer>, E: Pod> Borrow<[E]> for MappedBufferGuard<B, E> {
    fn borrow(&self) -> &[E] {
        let bytes = self.view.deref();
        bytemuck::cast_slice(bytes)
    }
}

pub(super) async fn read_mappable_buffer<E: SafeZero + Pod>(device: &GpuDevice, buffer: &GpuBuffer, sub_idx: Option<SubmissionIndex>) -> Result<Box<[E]>, BufferAsyncError> {
    let data = {
        let slice = buffer.slice(..);
        AsyncBufferView::new(slice, device, sub_idx).await?;
        let buf_view = slice.get_mapped_range();

        let mut data = calloc::<E>(buffer.size() as usize / size_of::<E>());
        let buf_slice = bytemuck::cast_slice::<u8, E>(&buf_view);
        data.copy_from_slice(buf_slice);
        
        data
    };

    Ok(data)
}