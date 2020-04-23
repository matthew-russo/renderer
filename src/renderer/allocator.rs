use crate::renderer::core::{RendererCore, run_with_device};
use crate::renderer::types::{Buffer, Uniform, Image, DescSetLayout, DescSet, DescSetWrite, Texture, TextureData};
use std::sync::{Arc, RwLock};
use hal::device::Device;
use hal::command::CommandBuffer;
use hal::queue::CommandQueue;
use hal::pool::CommandPool;
use hal::pso::DescriptorPool;

pub const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub(crate) enum DescriptorPoolType {
    Uniform,
    Texture,
}

pub(crate) trait Allocator<B: hal::Backend> {
    fn alloc_buffer<T>(&mut self, data: &[T], alignment: u64, min_size: u64, usage: hal::buffer::Usage, memory_properties: hal::memory::Properties) -> Buffer<B>
        where T: Copy,
              T: std::fmt::Debug;
    fn alloc_uniform<T>(&mut self, data: &[T], desc: DescSet<B>, binding: u32) -> Uniform<B>
        where T: Copy,
              T: std::fmt::Debug;
    fn alloc_image(&mut self, width: u32, height: u32, format: hal::format::Format, usage: hal::image::Usage, aspects: hal::format::Aspects) -> Image<B>;
    fn alloc_texture(&mut self, usage: hal::buffer::Usage, img_path: &String, sampler_desc: &hal::image::SamplerDesc, image_desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>) -> Texture<B>;
    fn alloc_desc_set_layout(&mut self, bindings: &[hal::pso::DescriptorSetLayoutBinding]) -> DescSetLayout<B>;
    fn alloc_desc_set(&mut self, pool_type: DescriptorPoolType, desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>) -> DescSet<B>;
}

pub(crate) struct GfxAllocator<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,

    image_desc_pool: Option<B::DescriptorPool>,
    uniform_desc_pool: Option<B::DescriptorPool>,
}

impl <B: hal::Backend> GfxAllocator<B> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        run_with_device(core, |device| {
            unsafe {
                let image_desc_pool = device
                    .create_descriptor_pool(
                        128,
                        &[
                            hal::pso::DescriptorRangeDesc {
                                ty: hal::pso::DescriptorType::Image {
                                    ty: hal::pso::ImageDescriptorType::Sampled {
                                        with_sampler: true,
                                    }
                                },
                                count: 128
                            }
                        ],
                        hal::pso::DescriptorPoolCreateFlags::empty()
                    )
                    .expect("Can't create descriptor pool");

                // TODO -> render graph, not static
                let uniform_desc_pool = device
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

                Self {
                    core: Arc::clone(core),
                    image_desc_pool: Some(image_desc_pool),
                    uniform_desc_pool: Some(uniform_desc_pool),
                }
            }
        })
    }

    fn transfer_image(&self,
                      image: &Image<B>,
                      mut image_upload_buffer: Buffer<B>,
                      image_extent: hal::image::Extent,
                      buffer_extent: hal::image::Extent)
    {
        let queue_family = self.core.read().unwrap().device.queue_group.family;
        run_with_device(&self.core, |device| {
            unsafe {
                let mut image_transferred_fence = device
                    .create_fence(false)
                    .expect("can't create fence");

                let mut staging_pool = device
                    .create_command_pool(
                        queue_family,
                        hal::pool::CommandPoolCreateFlags::empty(),
                    )
                    .expect("Can't create staging command pool");

                let cmds = self.transfer_image_cmd(&mut staging_pool,
                                                   &image,
                                                   &image_upload_buffer,
                                                   image_extent,
                                                   buffer_extent);

                self
                    .core
                    .write()
                    .unwrap()
                    .device
                    .queue_group
                    .queues[0]
                    .submit_without_semaphores(Some(&cmds), Some(&mut image_transferred_fence));

                image_upload_buffer.drop(device);
                device.destroy_command_pool(staging_pool);
                device.wait_for_fence(&image_transferred_fence, !0).unwrap();
            }
        });

    }

    fn transfer_image_cmd(&self,
                          command_pool: &mut B::CommandPool,
                          image: &Image<B>,
                          image_upload_buffer: &Buffer<B>,
                          image_extent: hal::image::Extent,
                          buffer_extent: hal::image::Extent)
        -> B::CommandBuffer
    {
        unsafe {
            let mut cmd_buffer = command_pool.allocate_one(hal::command::Level::Primary);
            cmd_buffer.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = hal::memory::Barrier::Image {
                states: (hal::image::Access::empty(), hal::image::Layout::Undefined)..(hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal),
                target: image.image.as_ref().unwrap(),
                families: None,
                range: COLOR_RANGE.clone(),
            };

            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::TOP_OF_PIPE..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                image_upload_buffer.buffer.as_ref().unwrap(),
                image.image.as_ref().unwrap(),
                hal::image::Layout::TransferDstOptimal,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: buffer_extent.width,
                    buffer_height: buffer_extent.height,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: hal::image::Extent {
                        width: image_extent.width,
                        height: image_extent.height,
                        depth: 1,
                    },
                }],
            );

            let image_barrier = hal::memory::Barrier::Image {
                states: (hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal)..(hal::image::Access::SHADER_READ, hal::image::Layout::ShaderReadOnlyOptimal),
                target: image.image.as_ref().unwrap(),
                families: None,
                range: COLOR_RANGE.clone(),
            };

            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::TRANSFER..hal::pso::PipelineStage::FRAGMENT_SHADER,
                hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();

            cmd_buffer
        }
    }

    fn find_memory_type(&self, mem_reqs: hal::memory::Requirements, props: hal::memory::Properties) -> hal::MemoryTypeId {
        self
            .core
            .read()
            .unwrap()
            .backend
            .adapter
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                mem_reqs.type_mask & (1 << id as u64) != 0
                    && mem_type.properties.contains(props)
            })
            .unwrap()
            .into()
    }

    fn calculate_stride<T>(alignment: u64) -> u64 {
        let data_stride = std::mem::size_of::<T>() as u64;
        if data_stride < alignment {
            alignment
        } else if data_stride % alignment == 0 {
            data_stride
        } else {
            let multiple = data_stride / alignment;
            alignment * (multiple + 1)
        }
    }
}

impl <B: hal::Backend> Allocator<B> for GfxAllocator<B> {
    fn alloc_buffer<T>(&mut self,
                    data: &[T],
                    alignment: u64,
                    min_size: u64,
                    usage: hal::buffer::Usage,
                    memory_properties: hal::memory::Properties)
        -> Buffer<B>
        where T: Copy,
              T: std::fmt::Debug
    {
        let stride = Self::calculate_stride::<T>(alignment);
        let mut upload_size = data.len() as u64 * stride;

        if upload_size < min_size {
            upload_size = min_size;
        }

        let mut buffer = run_with_device(&self.core, |device| {
            unsafe {
                device.create_buffer(upload_size, usage).unwrap()
            }
        });
        let mem_req = run_with_device(&self.core, |device| {
            unsafe {
                device.get_buffer_requirements(&buffer)
            }
        });

        // A note about performance: Using CPU_VISIBLE memory is convenient because it can be
        // directly memory mapped and easily updated by the CPU, but it is very slow and so should
        // only be used for small pieces of data that need to be updated very frequently. For something like
        // a vertex buffer that may be much larger and should not change frequently, you should instead
        // use a DEVICE_LOCAL buffer that gets filled by copying data from a CPU_VISIBLE staging buffer.
        let upload_type = self.find_memory_type(
            mem_req,
            memory_properties,
        );

        let memory = run_with_device(&self.core, |device| {
            unsafe {
                let memory = device
                    .allocate_memory(upload_type, mem_req.size)
                    .unwrap();

                device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();

                memory
            }
        });

        let mut buffer = Buffer::new(
            Some(buffer),
            Some(memory),
            false,
            mem_req.size,
            stride,
        );

        buffer.update_data(&self.core, 0, data);

        buffer
    }

    fn alloc_uniform<T>(&mut self,
                     data: &[T],
                     desc: DescSet<B>,
                     binding: u32) -> Uniform<B>
        where T: Copy,
              T: std::fmt::Debug
    {
        let alignment = self.core.read().unwrap().backend.adapter.limits.min_uniform_buffer_offset_alignment;
        let buffer = Some(self.alloc_buffer(
            &data,
            alignment,
            65536,
            hal::buffer::Usage::UNIFORM,
            hal::memory::Properties::CPU_VISIBLE,
        ));

        run_with_device(&self.core, |device| {
            desc.write(
                device,
                vec![DescSetWrite {
                    binding,
                    array_offset: 0,
                    descriptors: hal::pso::Descriptor::Buffer(
                        buffer.as_ref().unwrap().get_buffer(),
                        None..None,
                    )
                }]
            );
        });


        Uniform::new(
            buffer,
            Some(desc)
        )
    }

    fn alloc_image(&mut self,
        width: u32,
        height: u32,
        format: hal::format::Format,
        usage: hal::image::Usage,
        aspects: hal::format::Aspects) -> Image<B> {
        let mut image = run_with_device(&self.core, |device| {
            unsafe {
                device.create_image(
                         hal::image::Kind::D2(width, height, 1, 1),
                         1,
                         format,
                         hal::image::Tiling::Optimal,
                         usage,
                         hal::image::ViewCapabilities::empty()
                    )
                    .expect("failed to create image")
            }

        });

        let image_req = run_with_device(&self.core, |device| {
            unsafe { device.get_image_requirements(&image) }
        });

        let device_type = self.find_memory_type(image_req, hal::memory::Properties::DEVICE_LOCAL);

        let image_memory = run_with_device(&self.core, |device| {
            unsafe {
                let memory = device
                    .allocate_memory(device_type, image_req.size)
                    .expect("failed to allocate image memory");

                device
                    .bind_image_memory(&memory, 0, &mut image)
                    .expect("failed to bind memory to image");

                memory
            }
        });

        let image_view = run_with_device(&self.core, |device| {
            unsafe {
                device
                    .create_image_view(
                        &mut image,
                        hal::image::ViewKind::D2,
                        format,
                        hal::format::Swizzle::NO,
                        hal::image::SubresourceRange {
                            aspects,
                            levels: 0..1,
                            layers: 0..1,
                        }
                    )
                    .expect("failed to create image view")
            }
        });

        Image::new(
            Some(image),
            Some(image_view),
            Some(image_memory),
        )
    }

    fn alloc_texture(&mut self,
                     _usage: hal::buffer::Usage,
                     img_path: &String,
                     sampler_desc: &hal::image::SamplerDesc,
                     image_desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>) -> Texture<B> {
        let image_desc_set = self.alloc_desc_set(DescriptorPoolType::Texture, image_desc_set_layout);

        let row_alignment_mask = self.core.read().unwrap().backend.adapter.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let texture_data = TextureData::load(img_path, row_alignment_mask);

        let pixel_size = 4_usize;
        let row_pitch = (texture_data.width * pixel_size as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (texture_data.height * row_pitch) as u64;

        let image_upload_buffer = self.alloc_buffer(
            &texture_data.data,
            pixel_size as u64,
            upload_size,
            hal::buffer::Usage::TRANSFER_SRC,
            hal::memory::Properties::CPU_VISIBLE | hal::memory::Properties::COHERENT
        );

        let image = self.alloc_image(
            texture_data.width,
            texture_data.height,
            texture_data.format,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            hal::format::Aspects::COLOR,
        );

        self.transfer_image(&image,
                            image_upload_buffer,
                            hal::image::Extent {
                                width: texture_data.width,
                                height: texture_data.height,
                                depth: 1,
                            },
                            hal::image::Extent {
                                width: row_pitch / (pixel_size as u32),
                                height: texture_data.height,
                                depth: 1,
                            });

        let sampler = run_with_device(&self.core, |device| {
            let sampler = unsafe {
                device.create_sampler(sampler_desc).expect("can't create sampler")
            };

            image_desc_set.write(
                device,
                vec![
                    DescSetWrite {
                        binding: 0,
                        array_offset: 0,
                        descriptors: hal::pso::Descriptor::CombinedImageSampler(
                            image.image_view.as_ref().unwrap(),
                            hal::image::Layout::ShaderReadOnlyOptimal,
                            &sampler
                        )
                    }
                ]
            );

            sampler
        });

        Texture::new(
            image_desc_set,
            Some(sampler),
            image,
        )
    }

    fn alloc_desc_set_layout(&mut self, bindings: &[hal::pso::DescriptorSetLayoutBinding]) -> DescSetLayout<B> {
        let layout = run_with_device(&self.core, |device| {
            unsafe {
                device
                    .create_descriptor_set_layout(bindings, &[])
                    .expect("can't create descriptor set layout")
            }
        });

        DescSetLayout {
            layout: Some(layout),
        }
    }

    fn alloc_desc_set(&mut self, pool_type: DescriptorPoolType, desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>) -> DescSet<B> {
        let descriptor_pool = match pool_type {
            DescriptorPoolType::Uniform => self.uniform_desc_pool.as_mut().unwrap(),
            DescriptorPoolType::Texture => self.image_desc_pool.as_mut().unwrap(),
        };

        let descriptor_set = unsafe {
            descriptor_pool
                .allocate_set(desc_set_layout
                    .read()
                    .unwrap()
                    .layout
                    .as_ref()
                    .unwrap()
                )
                .unwrap()
        };

        DescSet {
            descriptor_set,
            desc_set_layout: Arc::clone(desc_set_layout),
        }
    }
}

impl <B: hal::Backend> Drop for GfxAllocator<B> {
    fn drop(&mut self) {
        let image_desc_pool = self.image_desc_pool.take().unwrap();
        let uniform_desc_pool = self.uniform_desc_pool.take().unwrap();
        run_with_device(&self.core, |device| {
            unsafe {
                device.destroy_descriptor_pool(image_desc_pool);
                device.destroy_descriptor_pool(uniform_desc_pool);
            }
        })
    }
}
