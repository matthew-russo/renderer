use std::sync::{Arc, RwLock};
use crate::utils::{any_as_u8_slice, data_path};
use crate::renderer::core::RendererCore;
use hal::device::Device;
use std::fs::File;
use std::io::BufReader;
use std::ops::DerefMut;

pub(crate) struct Buffer<B: hal::Backend> {
    pub buffer: Option<B::Buffer>,
    pub buffer_memory: Option<B::Memory>,
    pub memory_is_mapped: bool,
    pub size: u64,
    pub padded_stride: u64,
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

    pub(crate) fn drop(&mut self, device: &mut B::Device, ) {
        unsafe {
            device.destroy_buffer(self.buffer.take().unwrap());
            device.free_memory(self.buffer_memory.take().unwrap());
        }
    }

    pub fn get_buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().unwrap()
    }

    pub fn update_data<T>(&mut self, core: &Arc<RwLock<RendererCore<B>>>, offset: u64, data_source: &[T])
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
                .flat_map(|d| any_as_u8_slice(d, self.padded_stride as usize))
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

impl <B: hal::Backend> Uniform<B> {
    pub fn new(buffer: Option<Buffer<B>>, desc: Option<DescSet<B>>) -> Self {
        Self {
            buffer,
            desc,
        }
    }

    pub fn drop(&mut self, device: &mut B::Device) {
        self.buffer.as_mut().unwrap().drop(device);
        self.desc.as_mut().unwrap().drop(device);
    }
}

pub(crate) struct Image<B: hal::Backend> {
    pub image: Option<B::Image>,
    pub image_view: Option<B::ImageView>,
    pub image_memory: Option<B::Memory>,
}

impl<B: hal::Backend> Image<B> {
    pub fn new(image: Option<B::Image>,
               image_view: Option<B::ImageView>,
               image_memory: Option<B::Memory>)
        -> Self
    {
        Self {
            image,
            image_view,
            image_memory,
        }
    }

    pub fn drop(&mut self, device: &mut B::Device) {
        unsafe {
            device.destroy_image_view(self.image_view.take().unwrap());
            device.destroy_image(self.image.take().unwrap());
            device.free_memory(self.image_memory.take().unwrap());
        }
    }
}


pub(crate) struct Texture<B: hal::Backend> {
    pub desc_set: DescSet<B>,
    pub sampler: Option<B::Sampler>,
    pub image: Image<B>,
}

impl <B: hal::Backend> Texture<B> {
    pub fn new(desc_set: DescSet<B>,
               sampler: Option<B::Sampler>,
               image: Image<B>)
               -> Self
    {
        Self {
            desc_set,
            sampler,
            image,
        }
    }

    pub fn drop(&mut self, device: &mut B::Device) {
        unsafe {
            device.destroy_sampler(self.sampler.take().unwrap());
            self.image.drop(device);
        }
    }
}

pub(crate) struct TextureData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: hal::format::Format,
}

impl TextureData {
    pub fn load(img_path: &str, row_alignment_mask: u32) -> Self {
        let img_file = File::open(data_path(img_path)).unwrap();
        let img_reader = BufReader::new(img_file);
        let img = image::load(img_reader, image::JPEG)
            .unwrap()
            .to_rgba();

        let (width, height) = img.dimensions();

        // TODO -> duplicated in ImageState::new
        let image_stride = 4_usize;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;

        let size = (width * height) as usize * image_stride;
        let mut data: Vec<u8> = vec![0u8; size];

        for y in 0..height as usize {
            let row = &(*img)[y * (width as usize) * image_stride..(y + 1) * (width as usize) * image_stride];
            let start = y * row_pitch as usize;
            let count = width as usize * image_stride;
            let range = start..(start + count);
            data.splice(range, row.iter().map(|x| *x));
        }

        Self {
            width,
            height,
            data,
            format: hal::format::Format::Rgba8Srgb,
        }
    }
}

pub(crate) struct DescSetWrite<WI> {
    pub binding: hal::pso::DescriptorBinding,
    pub array_offset: hal::pso::DescriptorArrayIndex,
    pub descriptors: WI
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

    pub fn drop(&mut self, device: &mut B::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.layout.take().unwrap());
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
    pub(crate) fn write<'a, 'b: 'a, WI>(&'b self, device: &mut B::Device, writes: Vec<DescSetWrite<WI>>)
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

    pub fn drop(&mut self, device: &mut B::Device) {
         self.desc_set_layout.write().unwrap().deref_mut().drop(device);
    }
}
