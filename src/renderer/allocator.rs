use crate::utils::{any_as_u8_slice, data_path};
use crate::renderer::core::RendererCore;
use crate::renderer::types::{Buffer, Uniform, Image, DescSetLayout, DescSet};
use std::sync::{Arc, RwLock};
use std::fs::File;
use std::io::BufReader;
use hal::device::Device;
use hal::adapter::MemoryType;
use hal::queue::CommandQueue;
use hal::pso::DescriptorPool;

pub const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub(crate) trait Allocator<B: hal::Backend> {
    fn alloc_buffer(&mut self) -> Buffer<B>;
    fn alloc_uniform(&mut self) -> Uniform<B>;
    fn alloc_image(&mut self) -> Image<B>;
    fn alloc_desc_set_layout(&mut self) -> DescSetLayout<B>;
    fn alloc_desc_set(&mut self) -> DescSet<B>;
}

pub(crate) struct GfxAllocator<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,

    image_desc_pool: Option<B::DescriptorPool>,
    uniform_desc_pool: Option<B::DescriptorPool>,
    font_tex_desc_pool: Option<B::DescriptorPool>,
}

impl <B: hal::Backend> GfxAllocator<B> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        let image_desc_pool = core
            .read()
            .unwrap()
            .device
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
        let uniform_desc_pool = core
            .read()
            .unwrap()
            .device
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

        let font_tex_desc_pool = core
            .read()
            .unwrap()
            .device
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
            core: Arc::clone(core),
            image_desc_pool: Some(image_desc_pool),
            uniform_desc_pool: Some(uniform_desc_pool),
            font_tex_desc_pool: Some(font_tex_desc_pool),
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
    fn alloc_buffer(&mut self) -> Buffer<B> {
        let memory: B::Memory;
        let mut buffer: B::Buffer;

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

            let device = &core.read().unwrap().device.device;

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

        Buffer::new(
            Some(memory),
            Some(buffer),
            false,
            mem_req.size,
            padded_stride,
        )
    }

    fn alloc_uniform(&mut self,
                     data: &[T],
                     desc: DescSet<B>,
                     binding: u32) -> Uniform<B>
        where T: Copy,
              T: std::fmt::Debug
    {
        let buffer = Some(self.alloc_buffer(
            &core,
            &data,
            core.read().unwrap().backend.adapter.limits.min_uniform_buffer_offset_alignment,
            65536,
            hal::buffer::Usage::UNIFORM,
            &core.read().unwrap().backend.adapter.memory_types
        ));

        desc.write(
            &mut core.read().unwrap().device.device,
            vec![DescSetWrite {
                binding,
                array_offset: 0,
                descriptors: hal::pso::Descriptor::Buffer(
                    buffer.as_ref().unwrap().get_buffer(),
                    None..None,
                )
            }]
        );

        Uniform::new(
            buffer,
            Some(desc)
        )
    }

    fn alloc_image(&mut self,
                   _usage: hal::buffer::Usage,
                   img_path: &String,
                   sampler_desc: &hal::image::SamplerDesc,) -> Image<B> {
        let mut staging_pool = core
            .read()
            .unwrap()
            .device
            .device
            .create_command_pool(
                core
                    .read()
                    .unwrap()
                    .device
                    .queue_group
                    .family,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
            .expect("Can't create staging command pool");

        let image_desc_set = Self::create_set(self.image_desc_set_layout.as_ref().unwrap(), self.image_desc_pool.as_mut().unwrap());
        let row_alignment_mask = core.read().unwrap().backend.adapter.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_data = load_image_data(img_path, row_alignment_mask);

        let kind = hal::image::Kind::D2(image_data.width as hal::image::Size, image_data.height as hal::image::Size, 1, 1);
        let row_alignment_mask = core.read().unwrap().backend.adapter.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4_usize;
        let row_pitch = (image_data.width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (image_data.height * row_pitch) as u64;

        let mut image_upload_buffer = core.read().unwrap().device.device.create_buffer(upload_size, hal::buffer::Usage::TRANSFER_SRC).unwrap();
        let image_mem_reqs = core.read().unwrap().device.device.get_buffer_requirements(&image_upload_buffer);

        let upload_type = core
            .read()
            .unwrap()
            .backend
            .adapter
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
            let memory = core.read().unwrap().device.device.allocate_memory(upload_type, image_mem_reqs.size).unwrap();
            core.read().unwrap().device.device.bind_buffer_memory(&memory, 0, &mut image_upload_buffer).unwrap();
            let mapping = core.read().unwrap().device.device.map_memory(&memory, 0..upload_size).unwrap();

            // TODO -> duplicated in load_image_data
            for y in 0..image_data.height as usize {
                let row = &(*image_data.data)[y * (image_data.width as usize) * image_stride..(y+1) * (image_data.width as usize) * image_stride];
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    mapping.offset(y as isize * row_pitch as isize),
                    image_data.width as usize * image_stride
                );
            }

            core.read().unwrap().device.device.unmap_memory(&memory);
            memory
        };

        let mut image = core.read().unwrap().device.device.create_image(
            kind,
            1,
            image_data.format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            hal::image::ViewCapabilities::empty(),
        )
            .unwrap();

        let image_req = core.read().unwrap().device.device.get_image_requirements(&image);

        let device_type = core
            .read()
            .unwrap()
            .backend
            .adapter
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = core.read().unwrap().device.device.allocate_memory(device_type, image_req.size).unwrap();
        core.read().unwrap().device.device.bind_image_memory(&image_memory, 0, &mut image).unwrap();

        let mut transferred_image_fence = core.read().unwrap().device.device.create_fence(false).expect("Can't create fence");

        let mut cmd_buffer = command_pool.allocate_one(hal::command::Level::Primary);
        cmd_buffer.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);

        let image_barrier = hal::memory::Barrier::Image {
            states: (hal::image::Access::empty(), hal::image::Layout::Undefined)..(hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal),
            target: &image,
            families: None,
            range: COLOR_RANGE.clone(),
        };

        cmd_buffer.pipeline_barrier(
            hal::pso::PipelineStage::TOP_OF_PIPE..hal::pso::PipelineStage::TRANSFER,
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
                buffer_height: image_data.height as u32,
                image_layers: hal::image::SubresourceLayers {
                    aspects: hal::format::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                image_offset: hal::image::Offset { x: 0, y: 0, z: 0 },
                image_extent: hal::image::Extent {
                    width: image_data.width,
                    height: image_data.height,
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
            hal::pso::PipelineStage::TRANSFER..hal::pso::PipelineStage::FRAGMENT_SHADER,
            hal::memory::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.finish();

        core
            .read()
            .unwrap()
            .device
            .queue_group
            .queues[0]
            .submit_without_semaphores(Some(&cmd_buffer), Some(&mut transferred_image_fence));

        core
            .read()
            .unwrap()
            .device
            .device
            .wait_for_fence(&transferred_image_fence, !0)
            .unwrap();

        core.read().unwrap().device.device.destroy_buffer(image_upload_buffer);
        core.read().unwrap().device.device.free_memory(image_upload_memory);

        let image_view = core
            .read()
            .unwrap()
            .device
            .device
            .create_image_view(
                &image,
                hal::image::ViewKind::D2,
                image_data.format,
                hal::format::Swizzle::NO,
                COLOR_RANGE.clone(),
            ).unwrap();

        let sampler = core
            .read()
            .unwrap()
            .device
            .device
            .create_sampler(sampler_desc)
            .expect("Can't create sampler");

        desc_set.write(
            &mut core.read().unwrap().device.device,
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

        Image::new(
            desc_set,
            Some(sampler),
            Some(image),
            Some(image_view),
            Some(image_memory),
            Some(transferred_image_fence),
        )
    }

    fn alloc_desc_set_layout(&mut self) -> DescSetLayout<B> {
        let layout = unsafe {
            core
                .read()
                .unwrap()
                .device
                .device
                .create_descriptor_set_layout(bindings, &[])
        }.expect("Can't create descriptor set layout");

        Self {
            core: Arc::clone(core),
            layout: Some(layout),
        }
    }

    fn alloc_desc_set() -> DescSet<B> {
        unimplemented!()
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
    let img = image::load_image(img_reader, image::JPEG)
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
        format: hal::format::Format::Rgba8Srgb,
    };

    return image_data;
}

