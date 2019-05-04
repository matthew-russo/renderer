use std::fs;
use std::sync::{Arc, Mutex};
use std::io::{Cursor, Read};

use hal::format::{Format, AsFormat, ChannelType, Rgba8Srgb, Swizzle, Aspects};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags, VertexInputRate, Viewport};
use hal::queue::{Submission, family::QueueGroup};
use hal::pool::{CommandPool, RawCommandPool};
use hal::{
    Device as HalDevice,
    Backbuffer,
    DescriptorPool,
    FrameSync,
    Primitive,
    Swapchain,
    SwapchainConfig,
    Instance,
    PhysicalDevice,
    Surface,
    Adapter,
    MemoryType,
    CommandQueue,
    Graphics,
};
use hal::window::{Extent2D};

use image::load as load_image;

use crate::primitives::vertex::Vertex;

const DIMS: Extent2D = Extent2D { width: 1024,height: 768 };

const QUAD: [Vertex; 6] = [
    Vertex { in_position: [ -0.5, 0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [0.0, 1.0] },
    Vertex { in_position: [  0.5, 0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [1.0, 1.0] },
    Vertex { in_position: [  0.5,-0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [1.0, 0.0] },

    Vertex { in_position: [ -0.5, 0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [0.0, 1.0] },
    Vertex { in_position: [  0.5,-0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [1.0, 0.0] },
    Vertex { in_position: [ -0.5,-0.33, 0.0 ], in_color: [0.0, 0.0, 0.0], in_tex_coord: [0.0, 0.0] },
];

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub struct Renderer<B: hal::Backend> {
    surface: B::Surface,
    adapter: Adapter<B>,

    device: B::Device,
    queue_group: QueueGroup<B, Graphics>,

    swapchain: B::Swapchain,
    swapchain_format: Format,

    descriptor_pool: B::DescriptorPool,
    command_pool: CommandPool<B, Graphics>,

    descriptor_set_layout: B::DescriptorSetLayout,
    descriptor_set: B::DescriptorSet,

    vertex_buffer: B::Buffer,
    vertex_buffer_memory: B::Memory,

    image: B::Image,
    image_view: B::ImageView,
    image_memory: B::Memory,

    sampler: B::Sampler,

    render_pass: B::RenderPass,

    graphics_pipeline: B::GraphicsPipeline,
    pipeline_layout: B::PipelineLayout,

    backbuffer: Backbuffer<B>,
    framebuffers: Vec<B::Framebuffer>,
    frame_images: Vec<(B::Image, B::ImageView)>,

    frame_semaphore: B::Semaphore,
    frame_fence: B::Fence,

    viewport: Viewport,

    recreate_swapchain: bool,
    resize_dims: Extent2D,
}

impl<B: hal::Backend> Renderer<B> {
    pub fn initialize() -> Self {
        let mut events_loop = winit::EventsLoop::new();

        let wb = winit::WindowBuilder::new()
            .with_dimensions(winit::dpi::LogicalSize::new(
                DIMS.width as _,
                DIMS.height as _
            ))
            .with_title("matthew's spectacular rendering engine");

        #[cfg(not(feature = "gl"))]
        let (window, instance, mut adapters, mut surface) = {
            let window = wb.build(&events_loop).unwrap();
            let instance = back::Instance::create("matthew's spectacular rendering engine", 1);
            let surface = instance.create_surface(&window);
            let adapters = instance.enumerate_adapters();
            (window, instance, adapters, surface)
        };

        #[cfg(feature = "gl")]
        let (mut adapters, mut surface) = {
            let window = {
                let builder = back::config_context(back::glutin::ContextBuilder::new(), Rgba8Srgb::SELF, None).with_vsync(true);
                back::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
            };

            let surface = back::Surface::from_window(window);
            let adapters = surface.enumerate_adapters();
            (apaters, surface)
        };

        for adapter in &adapters {
            println!("adapter: {:?}", adapter.info);
        }

        // TODO -> pick the best adapter
        let mut adapter = adapters.remove(0);

        let memory_types = adapter.physical_device.memory_properties().memory_types;

        let (device, mut queue_group) = adapter
            .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
            .unwrap();

        let mut command_pool = unsafe {
            device.create_command_pool_typed(&queue_group, hal::pool::CommandPoolCreateFlags::empty())
        }.expect("Can't create command pool");

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);
        let mut descriptor_pool = Self::create_descriptor_pool(&device);

        let descriptor_set = unsafe { descriptor_pool.allocate_set(&descriptor_set_layout) }.unwrap();

        println!("Memory types: {:?}", memory_types);
        let mut frame_semaphore = device.create_semaphore().expect("Can't create semaphore");
        let mut frame_fence = device.create_fence(false).expect("Can't create fence");

        // vertx buffer
        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(&device, memory_types);

        // image, image view and sampler
        let (image, image_memory) = Self::create_image(&adapter, &device, memory_types, &command_pool, queue_group, &frame_fence);

        let image_view = Self::create_image_view(&device, &image);
        let sampler = Self::create_sampler(&device);

        //
        // DESCRIPTOR SETS WRITE
        //
        unsafe {
            device.write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Image(&image_view, hal::image::Layout::Undefined))
                },
                hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Sampler(&sampler))
                },
            ]);
        }

        //
        // SWAP CHAINS AND PRESENT MODES
        //
        let (caps, formats, _present_modes) = surface.compatability(&mut adapter.physical_device);
        println!("formats: {:?}", formats);
        let swapchain_format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = SwapchainConfig::from_caps(&caps, swapchain_format, DIMS);
        println!("swap_config: {:?}", swap_config);
        let extent = swap_config.extent.to_extent();

        let mut swap_buff =
            unsafe { device.create_swapchain(&mut surface, swap_config, None) }
                .expect("Can't create swapchain");

        let swapchain = swap_buff.0;
        let backbuffer = swap_buff.1;

        let render_pass = Self::create_render_pass(&device, swapchain_format);

        //
        // FRAMEBUFFER IMAGES
        //
        let (mut frame_images, mut framebuffers) = match backbuffer {
            Backbuffer::Images(images) => {
                let pairs = images
                    .into_iter()
                    .map(|image| unsafe {
                        let rtv = device
                            .create_image_view(
                                &image,
                                hal::image::ViewKind::D2,
                                swapchain_format,
                                Swizzle::NO,
                                COLOR_RANGE.clone(),
                            )
                            .unwrap();

                        (image, rtv)
                    })
                    .collect::<Vec<_>>();

                let fbos =  pairs
                    .iter()
                    .map(|&(_, ref rtv)| unsafe {
                        device
                            .create_framebuffer(&render_pass, Some(rtv), extent)
                            .unwrap()
                    })
                    .collect();

                (pairs, fbos)
            }

            Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
        };

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                std::iter::once(&descriptor_set_layout),
                &[(hal::pso::ShaderStageFlags::VERTEX, 0..8)],
            )
        }
        .expect("Can't create pipeline layout");

        let graphics_pipeline = Self::create_graphics_pipeline(&device, &pipeline_layout, &render_pass);

        let mut viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        let mut resize_dims = Extent2D {
            width: 0,
            height: 0,
        };

        Self {
            surface,
            adapter,

            device,
            queue_group,

            swapchain,
            swapchain_format,

            descriptor_pool,
            command_pool,

            descriptor_set_layout,
            descriptor_set,

            vertex_buffer,
            vertex_buffer_memory,

            image,
            image_view,
            image_memory,

            sampler,

            render_pass,

            graphics_pipeline,
            pipeline_layout,

            backbuffer,
            framebuffers,
            frame_images,

            frame_semaphore,
            frame_fence,

            viewport,

            recreate_swapchain: false,
            resize_dims,
        }
    }

    fn create_descriptor_set_layout(device: B::Device) -> B::DescriptorSetLayout {
        unsafe {
            device.create_descriptor_set_layout(
                &[
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: hal::pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false
                    },
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: hal::pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false
                    }
                ],
                &[]
            )
        }.expect("Can't create descriptor set layout")
    }

    fn create_descriptor_pool(device: B::Device) -> B::DescriptorPool {
        unsafe {
            device.create_descriptor_pool(
                1,
                &[
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::SampledImage,
                        count: 1
                    },
                    hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::Sampler,
                        count: 1
                    }
                ]
            )
        }.expect("Can't create descriptor pool")
    }

    fn create_vertex_buffer(device: B::Device, memory_types: Vec<MemoryType>) -> (B::Buffer, B::Memory) {
        let vertex_buffer_stride = std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer_len = QUAD.len() as u64 * vertex_buffer_stride;

        assert_ne!(vertex_buffer_len, 0);

        let mut vertex_buffer = unsafe { device.create_buffer(vertex_buffer_len, hal::buffer::Usage::VERTEX) }.unwrap();
        let vertex_buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                vertex_buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let vertex_buffer_memory = unsafe { device.allocate_memory(upload_type, vertex_buffer_req.size) }.unwrap();

        unsafe { device.bind_buffer_memory(&vertex_buffer_memory, 0, &mut vertex_buffer) }.unwrap();

        // TODO -> check transitions: read/write mapping and vertex buffer read
        unsafe {
            let mut vertices = device
                .acquire_mapping_writer::<Vertex>(&vertex_buffer_memory, 0..vertex_buffer_req.size)
                .unwrap();
            vertices[0..QUAD.len()].copy_from_slice(&QUAD);
            device.release_mapping_writer(vertices).unwrap();
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_image(adapter: &Adapter<B>,
                    device: &B::Device,
                    memory_types: Vec<MemoryType>,
                    command_pool: &CommandPool<B, Graphics>,
                    queue_group: CommandQueue<B, Graphics>,
                    frame_fence: &B::Fence)
        -> (B::Image, B::Memory)
    {
        let img_data = include_bytes!("data/textures/demo.jpg");
        let img = load_image(Cursor::new(&img_data[..]), image::JPEG)
            .unwrap()
            .to_rgba();

        let (width, height) = img.dimensions();
        let kind = hal::image::Kind::D2(width as hal::image::Size, height as hal::image::Size, 1, 1);
        let row_alignment_mask = adapter.physical_device.limits().min_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4_usize;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (height * row_pitch) as u64;

        let mut image_upload_buffer = unsafe { device.create_buffer(upload_size, hal::buffer::Usage::TRANSFER_SRC) }.unwrap();
        let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                image_mem_reqs.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let image_upload_memory = unsafe { device.allocate_memory(upload_type, image_mem_reqs.size) }.unwrap();

        unsafe {
            device.bind_buffer_memory(&image_upload_memory, 0, &mut image_upload_buffer)
        }.unwrap();

        unsafe {
            let mut data = device
                .acquire_mapping_writer::<u8>(&image_upload_memory, 0..image_mem_reqs.size)
                .unwrap();

            for y in 0..height as usize {
                let row = &(*img)[y * (width as usize) * image_stride..(y+1) * (width as usize) * image_stride];
                let dest_base = y * row_pitch as usize;
                data[dest_base..dest_base + row.len()].copy_from_slice(row);
            }

            device.release_mapping_writer(data).unwrap();
        }

        let mut image = unsafe {
            device.create_image(
                kind,
                1,
                Rgba8Srgb::SELF,
                hal::image::Tiling::Optimal,
                hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
                hal::image::ViewCapabilities::empty(),
            )
        }.unwrap();

        let image_req = unsafe { device.get_image_requirements(&image) };

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = unsafe { device.allocate_memory(device_type, image_req.size) }.unwrap();
        unsafe { device.bind_image_memory(&image_memory, 0, &mut image) }.unwrap();

        unsafe {
            let mut cmd_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
            cmd_buffer.begin();

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
                    buffer_height: height as u32,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: hal::image::Extent {
                        width,
                        height,
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

            queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(frame_fence));

            device
                .wait_for_fence(frame_fence, !0)
                .expect("Can't wait for fence");
        }

        unsafe {
            device.destroy_buffer(image_upload_buffer);
            device.free_memory(image_upload_memory);
        }

        (image, image_memory)
    }

    fn create_image_view(device: &B::Device, image: &B::Image) -> B::ImageView {
        unsafe {
            device.create_image_view(
                image,
                hal::image::ViewKind::D2,
                Rgba8Srgb::SELF,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            )
        }.unwrap()
    }

    fn create_sampler(device: &B::Device) -> B::Sampler {
        unsafe {
            device.create_sampler(hal::image::SamplerInfo::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp))
        }.expect("Can't create sampler")
    }

    fn create_render_pass(device: &B::Device, swapchain_format: Format) -> B::RenderPass {
        let attachment = hal::pass::Attachment {
            format: Some(swapchain_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined..hal::image::Layout::Present,
        };

        let subpass = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = hal::pass::SubpassDependency {
            passes: hal::pass::SubpassRef::External..hal::pass::SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: hal::image::Access::empty()..(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
        };

        unsafe { device.create_render_pass(&[attachment], &[subpass], &[dependency]) }.expect("Can't create render pass")
    }

    fn create_graphics_pipeline(device: &B::Device,
                                pipeline_layout: &B::PipelineLayout,
                                render_pass: &B::RenderPass)
        -> B::GraphicsPipeline {
        let vs_module = {
            let glsl = fs::read_to_string("data/shaders/standard.vert").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();

            unsafe { device.create_shader_module(&spirv) }.unwrap()
        };

        let fs_module = {
            let glsl = fs::read_to_string("data/shaders/standard.frag").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();

            unsafe { device.create_shader_module(&spirv) }.unwrap()
        };

        let pipeline = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint {
                    entry: "main",
                    module: &vs_module,
                    specialization: hal::pso::Specialization {
                        constants: &[hal::pso::SpecializationConstant { id: 0, range: 0..4 }],
                        data: unsafe { std::mem::transmute::<&f32, &[u8; 4]>(&0.8f32) },
                    },
                },
                hal::pso::EntryPoint {
                    entry: "main",
                    module: &fs_module,
                    specialization: hal::pso::Specialization::default(),
                }
            );

            let shader_entries = hal::pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let subpass = Subpass {
                index: 0,
                main_pass: render_pass,
            };

            let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                hal::pso::Rasterizer::FILL,
                pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc(
                hal::pso::ColorMask::ALL,
                hal::pso::BlendState::ALPHA,
            ));

            pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: VertexInputRate::Vertex,
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: hal::pso::Element {
                    format: Format::Rg32Float,
                    offset: 0,
                },
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: hal::pso::Element {
                    format: Format::Rg32Float,
                    offset: 8,
                },
            });

            unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
        };

        // clean up shader resources
        unsafe {
            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);
        }

        pipeline.unwrap()
    }

    pub fn draw_frame(&mut self, events_loop: &mut Arc<Mutex<winit::EventsLoop>>) {
        // TODO -> figure out event handling
        events_loop.lock().unwrap().poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                match event {
                    winit::WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    }
                        | winit::WindowEvent::CloseRequested => panic!("matthew's bad way of handling exit"),
                    winit::WindowEvent::Resized(dims) => {
                        println!("resized to: {:?}", dims);

                        #[cfg(feature = "gl")]
                        self.surface
                            .get_window()
                            .resize(dims.to_physical(self.surface.get_window().get_hidpi_factor()));

                        self.recreate_swapchain = true;
                        self.resize_dims.width = dims.width as u32;
                        self.resize_dims.height = dims.height as u32;
                    }
                    _ => (),
                }
            }
        });

        self.recreate_swapchain();

        let frame: hal::SwapImageIndex = unsafe {
            self.device.reset_fence(&self.frame_fence).unwrap();
            self.command_pool.reset();
            match self.swapchain.acquire_image(!0, FrameSync::Semaphore(&mut self.frame_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    self.recreate_swapchain = true;
                    return;
                }
            }
        };

        // Rendering
        let mut cmd_buffer = self.command_pool.acquire_command_buffer::<hal::command::OneShot>();
        unsafe {
            cmd_buffer.begin();

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&self.graphics_pipeline);
            cmd_buffer.bind_vertex_buffers(0, Some((&self.vertex_buffer, 0)));
            cmd_buffer.bind_graphics_descriptor_sets(&self.pipeline_layout, 0, Some(&self.descriptor_set), &[]);

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[frame as usize],
                    self.viewport.rect,
                    &[hal::command::ClearValue::Color(hal::command::ClearColor::Float([
                        0.8, 0.8, 0.8, 1.0,
                    ]))],
                );

                encoder.draw(0..6, 0..1);
            }

            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: Some((&self.frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: &[],
            };

            self.queue_group.queues[0].submit(submission, Some(&mut self.frame_fence));

            // TODO -> replace with semaphore
            self.device.wait_for_fence(&self.frame_fence, !0).unwrap();
            self.command_pool.free(Some(cmd_buffer));

            // present frame
            if let Err(_) = self.swapchain.present_nosemaphores(&mut self.queue_group.queues[0], frame) {
                self.recreate_swapchain = true;
            }
        }
    }

    fn recreate_swapchain(&mut self) {
        // recreate swapchain
        if self.recreate_swapchain {
            self.device.wait_idle().unwrap();

            let (caps, formats, _present_modes, _comp_alphas) = self.surface.compatibility(&mut self.adapter.physical_device);

            assert!(formats.iter().any(|fs| fs.contains(&self.swapchain_format)));

            let swap_config = SwapchainConfig::from_caps(&caps, self.swapchain_format, self.resize_dims);

            println!("swap_config: {:?}", swap_config);

            let extent = swap_config.extent.to_extent();

            let (new_swapchain, new_backbuffer) = unsafe {
                self.device.create_swapchain(&mut self.surface, swap_config, Some(self.swapchain))
            }.expect("Can't create swapchain");

            unsafe {
                // clean up the old framebuffers, images, and swapchain
                for framebuffer in self.framebuffers {
                    self.device.destroy_framebuffer(framebuffer);
                }

                for frame_image_arc in self.frame_images {
                    let frame_image_view = frame_image_arc.1;
                    self.device.destroy_image_view(frame_image_view);
                }
            }

            self.backbuffer = new_backbuffer;
            self.swapchain =new_swapchain;

            let (new_frame_images, new_framebuffers) = match self.backbuffer {
                Backbuffer::Images(images) => {
                    let pairs = images
                        .into_iter()
                        .map(|image| unsafe {
                            let rtv = self.device
                                .create_image_view(
                                    &image,
                                    hal::image::ViewKind::D2,
                                    self.swapchain_format.clone(),
                                    Swizzle::NO,
                                    COLOR_RANGE.clone(),
                                )
                                .unwrap();
                            (image, rtv)
                        })
                        .collect::<Vec<_>>();

                    let fbos = pairs
                        .iter()
                        .map(|&(_, ref rtv)| unsafe {
                            self.device
                                .create_framebuffer(&self.render_pass, Some(rtv), extent)
                                .unwrap()
                        })
                        .collect();

                    (pairs, fbos)
                },
                Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
            };

            self.framebuffers = new_framebuffers;
            self.frame_images = new_frame_images;
            self.viewport.rect.w = extent.width as _;
            self.viewport.rect.h = extent.height as _;
            self.recreate_swapchain = false;
        }
    }

    fn cleanup(self) {
        self.device.wait_idle().unwrap();

        unsafe {
            self.device.destroy_command_pool(self.command_pool.into_raw());
            self.device.destroy_descriptor_pool(self.descriptor_pool);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout);

            self.device.destroy_buffer(self.vertex_buffer);
            self.device.destroy_image(self.image);
            self.device.destroy_image_view(self.image_view);
            self.device.destroy_sampler(self.sampler);
            self.device.destroy_fence(self.frame_fence);
            self.device.destroy_semaphore(self.frame_semaphore);
            self.device.destroy_render_pass(self.render_pass);

            self.device.free_memory(self.vertex_buffer_memory);
            self.device.free_memory(self.image_memory);

            self.device.destroy_graphics_pipeline(self.graphics_pipeline);
            self.device.destroy_pipeline_layout(self.pipeline_layout);

            for framebuffer in self.framebuffers {
                self.device.destroy_framebuffer(framebuffer);
            }

            for (frame_image, frame_image_view) in self.frame_images {
                self.device.destroy_image(frame_image);
                self.device.destroy_image_view(frame_image_view);
            }

            self.device.destroy_swapchain(self.swapchain);
        }
    }
}