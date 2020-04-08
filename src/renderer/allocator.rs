use std::sync::{Arc, RwLock};
use std::fs::File;
use std::io::BufReader;
use ash::vk::MemoryType;
use crate::utils::{any_as_u8_slice, data_path};

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

trait Allocator<B: hal::Backend> {
    fn alloc_buffer() -> Buffer<B>;
    fn alloc_uniform() -> Uniform<B>;
    fn alloc_image() -> Image<B>;
    fn alloc_desc_set_layout() -> DescSetLayout<B>;
    fn alloc_desc_set() -> DescSet<B>;
}

struct GfxAllocator<B: hal::Backend> {
    device: Arc<RwLock<GfxDevice<B>>>,

    image_desc_pool: Option<B::DescriptorPool>,
    uniform_desc_pool: Option<B::DescriptorPool>,
    font_tex_desc_pool: Option<B::DescriptorPool>,
}

impl <B: hal::Backend> GfxAllocator<B> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        let mut image_desc_pool = core
            .device
            .read()
            .unwrap()
            .device
            .create_descriptor_pool(
                10,
                &[
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::Image {
                            ty: hal::pso::ImageDescriptorType::Sampled {
                                with_sampler: true,
                            }
                        },
                        count: 10
                    }
                ],
                hal::pso::DescriptorPoolCreateFlags::empty()
            )
            .expect("Can't create descriptor pool");

        // TODO -> render graph, not static
        let mut uniform_desc_pool = device_state
            .read()
            .unwrap()
            .device
            .create_descriptor_pool(
                2,
                &[
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::Buffer {
                            ty: hal::pso::BufferDescriptorType::Uniform,
                            format: hal::pso::BufferDescriptorFormat::Structured {
                                dynamic_offset: false,
                            }
                        },
                        count: 1
                    },
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::Buffer {
                            ty: hal::pso::BufferDescriptorType::Uniform,
                            format: hal::pso::BufferDescriptorFormat::Structured {
                                dynamic_offset: true,
                            }
                        },
                        count: 1
                    }
                ],
                hal::pso::DescriptorPoolCreateFlags::empty()
            )
            .expect("Can't create descriptor pool");

        let mut font_tex_desc_pool = device_state
            .read()
            .unwrap()
            .device
            .create_descriptor_pool(
                10,
                &[
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::Sampler,
                        count: 10
                    }
                ],
                hal::pso::DescriptorPoolCreateFlags::empty()
            )
            .expect("Can't create descriptor pool");

        Self {

        }
    }

    fn create_set(desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>, descriptor_pool: &mut B::DescriptorPool) -> DescSet<B> {
        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(desc_set_layout.read().unwrap().layout.as_ref().unwrap())
        }.unwrap();

        DescSet {
            descriptor_set,
            desc_set_layout: desc_set_layout.clone()
        }
    }
}

impl <B: hal::Backend> Allocator<B> for GfxAllocator<B> {
    fn alloc_buffer() -> Buffer<B> {
        unimplemented!()
    }

    fn alloc_uniform() -> Uniform<B> {
        unimplemented!()
    }

    fn alloc_image() -> Image<B> {
        unimplemented!()
    }

    fn alloc_desc_set() -> DescSet<B> {

    }
}

struct Buffer<B: hal::Backend> {
    buffer: Option<B::Buffer>,
    buffer_memory: Option<B::Memory>,
    memory_is_mapped: bool,
    size: u64,
    padded_stride: u64,
    device_state: Arc<RwLock<DeviceState<B>>>,
}

impl<B: hal::Backend> Buffer<B> {
    fn get_buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().unwrap()
    }

    unsafe fn new<T: Sized>(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        data_source: &[T],
        alignment: u64,
        min_size: u64,
        usage: hal::buffer::Usage,
        memory_types: &[MemoryType]
    ) -> Self
        where T: Copy,
              T: std::fmt::Debug
    {
        let memory: B::Memory;
        let mut buffer: B::Buffer;
        let size: u64;

        let data_stride = std::mem::size_of::<T>() as u64;
        let padded_stride = if data_stride < alignment {
            alignment
        } else if data_stride % alignment == 0 {
            data_stride
        } else {
            let multiple = data_stride / alignment;
            alignment * (multiple + 1)
        };
        let mut upload_size = data_source.len() as u64 * padded_stride;

        if upload_size < min_size {
            upload_size = min_size;
        }

        {
            let device = &device_state.read().unwrap().device;

            buffer = device.create_buffer(upload_size, usage).unwrap();
            let mem_req = device.get_buffer_requirements(&buffer);

            // A note about performance: Using CPU_VISIBLE memory is convenient because it can be
            // directly memory mapped and easily updated by the CPU, but it is very slow and so should
            // only be used for small pieces of data that need to be updated very frequently. For something like
            // a vertex buffer that may be much larger and should not change frequently, you should instead
            // use a DEVICE_LOCAL buffer that gets filled by copying data from a CPU_VISIBLE staging buffer.
            let upload_type = memory_types
                .iter()
                .enumerate()
                .position(|(id, mem_type)| {
                    mem_req.type_mask & (1 << id) != 0
                        && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                })
                .unwrap()
                .into();

            memory = device.allocate_memory(upload_type, mem_req.size).unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            size = mem_req.size;

            // TODO: check transitions: read/write mapping and vertex buffer read
            {
                let mapping = device.map_memory(&memory, 0..size).unwrap();

                let data_as_bytes = data_source
                    .iter()
                    .flat_map(|ubo| { any_as_u8_slice(ubo, padded_stride as usize) })
                    .collect::<Vec<u8>>();
                std::ptr::copy_nonoverlapping(
                    data_as_bytes.as_ptr(),
                    mapping.offset(0),
                    size as usize
                );

                device.unmap_memory(&memory);
            }
        }

        Self {
            buffer_memory: Some(memory),
            buffer: Some(buffer),
            memory_is_mapped: false,
            size,
            padded_stride,
            device_state: device_state.clone(),
        }
    }

    fn update_data<T>(&mut self, offset: u64, data_source: &[T])
        where T: Copy,
              T: std::fmt::Debug
    {
        let device = &self.device_state.read().unwrap().device;
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

impl<B: hal::Backend> Drop for Buffer<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_buffer(self.buffer.take().unwrap());
            device.free_memory(self.buffer_memory.take().unwrap());
        }
    }
}

pub(crate) struct Uniform<B: hal::Backend> {
    pub buffer: Option<BufferState<B>>,
    pub desc: Option<DescSet<B>>,
}

impl<B: hal::Backend> Uniform<B> {
    unsafe fn new<T>(
        adapter_state: &AdapterState<B>,
        device_state: &Arc<RwLock<DeviceState<B>>>,
        data: &[T],
        desc: DescSet<B>,
        binding: u32
    ) -> Self
        where T: Copy,
              T: std::fmt::Debug
    {
        let buffer = BufferState::new(
            &device_state,
            &data,
            adapter_state.limits.min_uniform_buffer_offset_alignment,
            65536,
            hal::buffer::Usage::UNIFORM,
            &adapter_state.memory_types
        );
        let buffer = Some(buffer);

        desc.write(
            &mut device_state.write().unwrap().device,
            vec![DescSetWrite {
                binding,
                array_offset: 0,
                descriptors: hal::pso::Descriptor::Buffer(
                    buffer.as_ref().unwrap().get_buffer(),
                    None..None,
                )
            }]
        );

        Self {
            buffer,
            desc: Some(desc)
        }
    }
}

pub(crate) struct Image<B: hal::Backend> {
    pub desc_set: DescSet<B>,
    pub sampler: Option<B::Sampler>,
    pub image: Option<B::Image>,
    pub image_view: Option<B::ImageView>,
    pub image_memory: Option<B::Memory>,
    pub transferred_image_fence: Option<B::Fence>,
    pub device_state: Arc<RwLock<DeviceState<B>>>
}

// TODO -> refactor this --
//      - pass image data in,
//      - take create_image function into account
//
impl<B: hal::Backend> Image<B> {
    pub unsafe fn new(
        _usage: hal::buffer::Usage,
        img_path: &String,
        sampler_desc: &hal::image::SamplerDesc,
    ) -> Self {
        let mut staging_pool = self.device_state
            .read()
            .unwrap()
            .device
            .create_command_pool(
                self.device_state
                    .read()
                    .unwrap()
                    .queue_group
                    .family,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
            .expect("Can't create staging command pool");

        let image_desc_set = Self::create_set(self.image_desc_set_layout.as_ref().unwrap(), self.image_desc_pool.as_mut().unwrap());
        let row_alignment_mask = self.backend_state.adapter_state.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_data = load_image_data(img_path, row_alignment_mask);

        let kind = hal::image::Kind::D2(img_data.width as hal::image::Size, img_data.height as hal::image::Size, 1, 1);
        let row_alignment_mask = adapter_state.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4_usize;
        let row_pitch = (img_data.width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (img_data.height * row_pitch) as u64;

        let mut image_upload_buffer = device_state.read().unwrap().device.create_buffer(upload_size, hal::buffer::Usage::TRANSFER_SRC).unwrap();
        let image_mem_reqs = device_state.read().unwrap().device.get_buffer_requirements(&image_upload_buffer);

        let upload_type = adapter_state
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                image_mem_reqs.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(
                    hal::memory::Properties::CPU_VISIBLE | hal::memory::Properties::COHERENT)
            })
            .unwrap()
            .into();

        let image_upload_memory = {
            let memory = device_state.read().unwrap().device.allocate_memory(upload_type, image_mem_reqs.size).unwrap();
            device_state.read().unwrap().device.bind_buffer_memory(&memory, 0, &mut image_upload_buffer).unwrap();
            let mapping = device_state.read().unwrap().device.map_memory(&memory, 0..upload_size).unwrap();

            // TODO -> duplicated in load_image_data
            for y in 0..img_data.height as usize {
                let row = &(*img_data.data)[y * (img_data.width as usize) * image_stride..(y+1) * (img_data.width as usize) * image_stride];
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    img_data.width as usize * image_stride
                );
            }

            device_state.read().unwrap().device.unmap_memory(&memory);
            memory
        };

        let mut image = device_state.read().unwrap().device.create_image(
            kind,
            1,
            img_data.format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            hal::image::ViewCapabilities::empty(),
        )
            .unwrap();

        let image_req = device_state.read().unwrap().device.get_image_requirements(&image);

        let device_type = adapter_state
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = device_state.read().unwrap().device.allocate_memory(device_type, image_req.size).unwrap();
        device_state.read().unwrap().device.bind_image_memory(&image_memory, 0, &mut image).unwrap();

        let mut transferred_image_fence = device_state.read().unwrap().device.create_fence(false).expect("Can't create fence");

        let mut cmd_buffer = command_pool.allocate_one(hal::command::Level::Primary);
        cmd_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

        let image_barrier = hal::memory::Barrier::Image {
            states: (hal::image::Access::empty(), hal::image::Layout::Undefined)..(hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal),
            target: &image,
            families: None,
            range: COLOR_RANGE.clone(),
        };

        cmd_buffer.pipeline_barrier(
            PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
            hal::memory::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.copy_buffer_to_image(
            &image_upload_buffer,
            &image,
            hal::image::Layout::TransferDstOptimal,
            &[hal::command::BufferImageCopy {
                buffer_offset: 0,
                buffer_width: row_pitch / (image_stride as u32),
                buffer_height: img_data.height as u32,
                image_layers: hal::image::SubresourceLayers {
                    aspects: Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                image_offset: hal::image::Offset { x: 0, y: 0, z: 0 },
                image_extent: hal::image::Extent {
                    width: img_data.width,
                    height: img_data.height,
                    depth: 1,
                },
            }],
        );

        let image_barrier = hal::memory::Barrier::Image {
            states: (hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal)..(hal::image::Access::SHADER_READ, hal::image::Layout::ShaderReadOnlyOptimal),
            target: &image,
            families: None,
            range: COLOR_RANGE.clone(),
        };

        cmd_buffer.pipeline_barrier(
            PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
            hal::memory::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.finish();

        device_state
            .write()
            .unwrap()
            .queue_group
            .queues[0]
            .submit_without_semaphores(Some(&cmd_buffer), Some(&mut transferred_image_fence));

        device_state
            .write()
            .unwrap()
            .device
            .wait_for_fence(&transferred_image_fence, !0)
            .unwrap();

        device_state.read().unwrap().device.destroy_buffer(image_upload_buffer);
        device_state.read().unwrap().device.free_memory(image_upload_memory);

        let image_view = device_state.read().unwrap().device
            .create_image_view(
                &image,
                hal::image::ViewKind::D2,
                img_data.format,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            ).unwrap();

        let sampler = device_state.read().unwrap().device
            .create_sampler(sampler_desc)
            .expect("Can't create sampler");

        desc_set.write(
            &mut device_state.write().unwrap().device,
            vec![
                DescSetWrite {
                    binding: 0,
                    array_offset: 0,
                    descriptors: hal::pso::Descriptor::CombinedImageSampler(
                        &image_view,
                        hal::image::Layout::ShaderReadOnlyOptimal,
                        &sampler
                    )
                }
            ]
        );

        Self {
            desc_set,
            sampler: Some(sampler),
            image: Some(image),
            image_view: Some(image_view),
            image_memory: Some(image_memory),
            transferred_image_fence: Some(transferred_image_fence),
            device_state: device_state.clone()
        }
    }

    pub fn wait_for_transfer_completion(&self) {
        let readable_desc_set = self.desc_set.desc_set_layout.read().unwrap();
        let device = &readable_desc_set.device_state.read().unwrap().device;
        unsafe {
            device
                .wait_for_fence(&self.transferred_image_fence.as_ref().unwrap(), !0)
                .unwrap();
        }
    }
}

struct ImageData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: hal::format::Format,
}

fn load_image_data(img_path: &str, row_alignment_mask: u32) -> ImageData {
    let img_file = File::open(data_path(img_path)).unwrap();
    let img_reader = BufReader::new(img_file);
    let img = load_image(img_reader, image::JPEG)
        .unwrap()
        .to_rgba();

    let (width, height) = img.dimensions();

    // TODO -> duplicated in ImageState::new
    let image_stride = 4_usize;
    let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;

    let size = (width * height) as usize * image_stride;
    let mut data: Vec<u8> = vec![0u8; size];

    for y in 0..height as usize {
        let row = &(*img)[y * (width as usize) * image_stride..(y+1) * (width as usize) * image_stride];
        let start = y * row_pitch as usize;
        let count = width as usize * image_stride;
        let range = start..(start + count);
        data.splice(range, row.iter().map(|x| *x));
    }

    let image_data = ImageData {
        width,
        height,
        data,
        format: Rgba8Srgb::SELF,
    };

    return image_data;
}

impl<B: hal::Backend> Drop for Image<B> {
    fn drop(&mut self) {
        unsafe {
            let readable_desc_set_layout = self.desc_set.desc_set_layout.read().unwrap();
            let device = &readable_desc_set_layout.device_state.read().unwrap().device;

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

struct DescSet<B: hal::Backend> {
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

struct DescSetWrite<WI> {
    binding: hal::pso::DescriptorBinding,
    array_offset: hal::pso::DescriptorArrayIndex,
    descriptors: WI
}

struct DescSetLayout<B: hal::Backend> {
    layout: Option<B::DescriptorSetLayout>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> DescSetLayout<B> {
    fn new(device_state: &Arc<RwLock<DeviceState<B>>>, bindings: Vec<hal::pso::DescriptorSetLayoutBinding>) -> Self {
        let layout = unsafe {
            device_state
                .read()
                .unwrap()
                .device
                .create_descriptor_set_layout(bindings, &[])
        }.expect("Can't create descriptor set layout");

        Self {
            layout: Some(layout),
            device_state: device_state.clone()
        }
    }
}

impl<B: hal::Backend> Drop for DescSetLayout<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_descriptor_set_layout(self.layout.take().unwrap());
        }
    }
}



