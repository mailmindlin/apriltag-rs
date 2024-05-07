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
            Some(sub_idx) => wgpu::MaintainBase::WaitForSubmissionIndex(sub_idx),
            None => wgpu::MaintainBase::Poll,
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

pub(super) struct MappedBufferGuard<'a, B: Borrow<GpuBuffer>, E: AnyBitPattern> {
    buffer: B,
    view: BufferView<'a>,
    element: PhantomData<E>,
}

impl<'a, B: Borrow<GpuBuffer>, E: AnyBitPattern> MappedBufferGuard<'a, B, E> {
    fn new(buffer: B, view: BufferView<'a>) -> Self {
        Self {
            buffer,
            view,
            element: PhantomData,
        }
    }
}
impl<B: Borrow<GpuBuffer>, E: Pod> Borrow<[E]> for MappedBufferGuard<'_, B, E> {
    fn borrow(&self) -> &[E] {
        let bytes = self.view.deref();
        bytemuck::cast_slice(bytes)
    }
}

// impl<E: AnyBitPattern + Clone> ToOwned for MappedBufferGuard<'_, E> {
//     type Owned = Vec<E>;

//     fn to_owned(&self) -> Self::Owned {
//         let elems = <Self as Deref<[E]>>::deref(self);
//         elems.to_owned()
//     }
// }
// impl<E: Pod> Deref for MappedBufferGuard<'_, E> {
//     type Target = [E];
//     fn deref(&self) -> &Self::Target {
//         let bytes = self.view.deref();
//         bytemuck::cast_slice(bytes)
//     }
// }
// impl<E: Pod> AsRef<[E]> for MappedBufferGuard<'_, E> {
//     fn as_ref(&self) -> &[E] {
//         self.deref()
//     }
// }

// impl<E: AnyBitPattern, B: Borrow<GpuBuffer>> Drop for MappedBufferGuard<'_, B, E> {
//     fn drop(&mut self) {
//         self.buffer.borrow().unmap();
//     }
// }

// pub(super) async fn map_buffer<'a, E: Pod, B: Borrow<GpuBuffer> + 'a>(device: &'a GpuDevice, buffer_owner: B, sub_idx: Option<SubmissionIndex>) -> Result<MappedBufferGuard<'a, B, E>, BufferAsyncError> {
//     let buffer: &GpuBuffer = buffer_owner.borrow();
//     let x = Pin::new(buffer);
//     let buf_view = AsyncBufferView::new(x.slice(..), device, sub_idx).await?;

//     Ok(MappedBufferGuard::new(buffer_owner, buf_view))
// }

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