use crate::utils::{any_as_u8_slice};
use std::sync::{Arc, RwLock};
use hal::adapter::MemoryType;
use hal::device::Device;

pub(crate) struct Buffer<B: hal::Backend> {
    buffer: Option<B::Buffer>,
    buffer_memory: Option<B::Memory>,
    memory_is_mapped: bool,
    size: u64,
    padded_stride: u64,
}

impl<B: hal::Backend> Buffer<B> {
    pub fn new(buffer: Option<B::Buffer>,
               buffer_memory: Option<B::Memory>,
               memory_is_mapped: bool,
               size: u64,
               padded_stride: u64,) -> Self {
        Self {
            buffer,
            buffer_memory,
            memory_is_mapped,
            size,
            padded_stride,
        }
    }

    pub fn get_buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().unwrap()
    }

    pub fn update_data<T>(&mut self, core: &Arc<RwLock<B>>, offset: u64, data_source: &[T])
        where T: Copy,
              T: std::fmt::Debug
    {
        let device = &core.read().unwrap().device.device;
        let upload_size = data_source.len() as u64 * self.padded_stride;

        assert!(offset + upload_size <= self.size);

        unsafe {
            let mapping = device.map_memory(self.buffer_memory.as_ref().unwrap(), offset..upload_size).unwrap();

            let data_as_bytes = data_source
                .iter()
                .flat_map(|ubo| any_as_u8_slice(ubo, self.padded_stride as usize))
                .collect::<Vec<u8>>();
            std::ptr::copy_nonoverlapping(
                data_as_bytes.as_ptr(),
                mapping.offset(0),
                upload_size as usize
            );

            device.unmap_memory(self.buffer_memory.as_ref().unwrap());
        }
    }
}

pub(crate) struct Uniform<B: hal::Backend> {
    pub buffer: Option<Buffer<B>>,
    pub desc: Option<DescSet<B>>,
}

pub(crate) struct Image<B: hal::Backend> {
    pub desc_set: DescSet<B>,
    pub sampler: Option<B::Sampler>,
    pub image: Option<B::Image>,
    pub image_view: Option<B::ImageView>,
    pub image_memory: Option<B::Memory>,
    pub transferred_image_fence: Option<B::Fence>,
}

impl<B: hal::Backend> Image<B> {
    pub fn new(desc_set: DescSet<B>,
               sampler: Option<B::Sampler>,
               image: Option<B::Image>,
               image_view: Option<B::ImageView>,
               image_memory: Option<B::Memory>,
               transferred_image_fence: Option<B::Fence>)
        -> Self
    {
        Self {
            desc_set,
            sampler,
            image,
            image_view,
            image_memory,
            transferred_image_fence,
        }
    }

    pub fn wait_for_transfer_completion(&self) {
        let readable_desc_set = self.desc_set.desc_set_layout.read().unwrap();
        let device = &readable_desc_set.core.read().unwrap().device.device;
        unsafe {
            device
                .wait_for_fence(&self.transferred_image_fence.as_ref().unwrap(), !0)
                .unwrap();
        }
    }
}


pub(crate) struct DescSetWrite<WI> {
    binding: hal::pso::DescriptorBinding,
    array_offset: hal::pso::DescriptorArrayIndex,
    descriptors: WI
}

pub(crate) struct DescSetLayout<B: hal::Backend> {
    pub layout: Option<B::DescriptorSetLayout>,
}

impl<B: hal::Backend> DescSetLayout<B> {
    fn new(layout: Option<B::DescriptorSetLayout>) -> Self {
        Self {
            layout,
        }
    }
}

pub(crate) struct DescSet<B: hal::Backend> {
    pub descriptor_set: B::DescriptorSet,
    pub desc_set_layout: Arc<RwLock<DescSetLayout<B>>>,
}
// vec![
//     hal::pso::DescriptorSetWrite {
//         set: &descriptor_set,
//         binding: 0,
//         array_offset: 0,
//         descriptors: Some(hal::pso::Descriptor::Image(&image_view, hal::image::Layout::Undefined))
//     },
//     hal::pso::DescriptorSetWrite {
//         set: &descriptor_set,
//         binding: 1,
//         array_offset: 0,
//         descriptors: Some(hal::pso::Descriptor::Sampler(&sampler))
//     },
// ]

impl<B: hal::Backend> DescSet<B> {
    fn write<'a, 'b: 'a, WI>(&'b self, device: &mut B::Device, writes: Vec<DescSetWrite<WI>>)
        where
            WI: std::borrow::Borrow<hal::pso::Descriptor<'a, B>>
    {
        let descriptor_set_writes = writes
            .into_iter()
            .map(|dsw| hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: dsw.binding,
                array_offset: dsw.array_offset,
                descriptors: Some(dsw.descriptors)
            });

        unsafe {
            device.write_descriptor_sets(descriptor_set_writes);
        }
    }
}

impl<B: hal::Backend> Drop for Image<B> {
    fn drop(&mut self) {
        unsafe {
            let readable_desc_set_layout = self.desc_set.desc_set_layout.read().unwrap();
            let device = &readable_desc_set_layout.core.read().unwrap().device.device;

            let fence = self.transferred_image_fence.take().unwrap();
            device.wait_for_fence(&fence, !0).unwrap();
            device.destroy_fence(fence);

            device.destroy_sampler(self.sampler.take().unwrap());
            device.destroy_image_view(self.image_view.take().unwrap());
            device.destroy_image(self.image.take().unwrap());
            device.free_memory(self.image_memory.take().unwrap());
        }
    }
}

impl<B: hal::Backend> Drop for Buffer<B> {
    fn drop(&mut self) {
        let device = &self.core.read().unwrap().device.device;
        unsafe {
            device.destroy_buffer(self.buffer.take().unwrap());
            device.free_memory(self.buffer_memory.take().unwrap());
        }
    }
}

impl<B: hal::Backend> Drop for DescSetLayout<B> {
    fn drop(&mut self) {
        let device = &self.core.read().unwrap().device.device;
        unsafe {
            device.destroy_descriptor_set_layout(self.layout.take().unwrap());
        }
    }
}

