use std::fs;
use std::sync::{Arc, RwLock};
use std::io::{Cursor, Read};

use hal::format::{Format, AsFormat, ChannelType, Rgba8Srgb, Swizzle, Aspects};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags, VertexInputRate, Viewport};
use hal::queue::{Submission, family::QueueGroup};
use hal::pool::{CommandPool};
use hal::window::{Extent2D};
use hal::{
    Device as HalDevice,
    DescriptorPool,
    Primitive,
    Swapchain,
    SwapchainConfig,
    Instance,
    PhysicalDevice,
    Surface,
    Adapter,
    MemoryType,
    Graphics,
    Limits,
};

use image::load as load_image;

use crate::primitives::vertex::Vertex;
use crate::events::event_handler::EventHandler;

const DIMS: Extent2D = Extent2D { width: 1024,height: 768 };

const QUAD: [Vertex; 6] = [
    Vertex { in_position: [ -0.5, 0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [0.0, 1.0] },
    Vertex { in_position: [  0.5, 0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [1.0, 1.0] },
    Vertex { in_position: [  0.5,-0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [1.0, 0.0] },

    Vertex { in_position: [ -0.5, 0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [0.0, 1.0] },
    Vertex { in_position: [  0.5,-0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [1.0, 0.0] },
    Vertex { in_position: [ -0.5,-0.33, 0.0 ], in_color: [1.0, 1.0, 1.0], in_tex_coord: [0.0, 0.0] },
];

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

struct AdapterState<B: hal::Backend> {
    adapter: Option<Adapter<B>>,
    memory_types: Vec<MemoryType>,
    limits: Limits,
}

impl<B: hal::Backend> AdapterState<B> {
    fn new(adapters: &mut Vec<Adapter<B>>) -> Self {
        match Self::pick_best_adapter(adapters) {
            Some(adapter) => Self::create_adapter_state(adapter),
            None => panic!("Failed to pick an adapter")
        }
    }

    fn pick_best_adapter(adapters: &mut Vec<Adapter<B>>) -> Option<Adapter<B>> {
        if adapters.is_empty() {
           return None; 
        }
       
        // TODO -> smarter adapter selection
        return Some(adapters.remove(0));
    }

    fn create_adapter_state(adapter: Adapter<B>) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();
    
        Self {
            adapter: Some(adapter),
            memory_types,
            limits
        }
    }
}

pub struct BackendState<B: hal::Backend> {
    surface: B::Surface,
    adapter_state: AdapterState<B>,

    #[cfg(any(feature = "vulkan", feature = "dx11", feature = "dx12", feature = "metal"))]
    #[allow(dead_code)]
    window: winit::Window,
}

#[cfg(not(any(feature="gl", feature="dx12", feature="vulkan", feature="metal")))]
pub fn create_backend(_window_state: &WindowState) -> (BackendState<back::Backend>, ()) {
    panic!("You must specify one of the valid backends using --features=<backend>, with \"gl\", \"dx12\", \"vulkan\", and \"metal\" being valid backends.");
}

#[cfg(feature="gl")]
pub fn create_backend(window_state: &WindowState) -> (BackendState<back::Backend>, ()) {
    let (mut adapters, mut surface) = {
        let window = {
            let builder = back::config_context(back::glutin::ContextBuilder::new(), Rgba8Srgb::SELF, None).with_vsync(true);
            back::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
        };

        let surface = back::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (apaters, surface)
    };

    let backend_state = BackendState {
        surface,
        adapter_state: AdapterState::new(adapters),
    };

    (backend_state, ())
}

#[cfg(any(feature="dx12", feature="vulkan", feature="metal"))]
pub fn create_backend(window_state: &mut WindowState) -> (BackendState<back::Backend>, back::Instance) {
    let window = window_state
        .window_builder
        .take()
        .unwrap()
        .build(&window_state.events_loop)
        .unwrap();

    let instance = back::Instance::create("matthew's spectacular rendering engine", 1);
    let surface = instance.create_surface(&window);
    let mut adapters = instance.enumerate_adapters();

    let backend_state = BackendState {
        surface,
        adapter_state: AdapterState::new(&mut adapters),
        window
    };

    (backend_state, instance)
}

pub struct WindowState {
    events_loop: winit::EventsLoop,
    window_builder: Option<winit::WindowBuilder>,
}

impl WindowState {
    pub fn new() -> Self {
        let events_loop = winit::EventsLoop::new();

        let wb = winit::WindowBuilder::new()
            .with_dimensions(winit::dpi::LogicalSize::new(
                DIMS.width as _,
                DIMS.height as _
            ))
            .with_title("matthew's spectacular rendering engine");

        Self {
            events_loop,
            window_builder: Some(wb)
        }
    }
}

struct DeviceState<B: hal::Backend> {
    device: B::Device,
    physical_device: B::PhysicalDevice,
    queue_group: QueueGroup<B, Graphics>,
}

impl<B: hal::Backend> DeviceState<B> {
    fn new(adapter: Adapter<B>, surface: &Surface<B>) -> Self {
        let (device, queue_group) = adapter
            .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
            .unwrap();

        Self {
            device,
            physical_device: adapter.physical_device,
            queue_group
        }
    }
}

struct RenderPassState<B: hal::Backend> {
    render_pass: Option<B::RenderPass>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> RenderPassState<B> {
    fn new(device_state: &Arc<RwLock<DeviceState<B>>>, swapchain_state: &SwapchainState<B>) -> Self {
        let device = &device_state
            .read()
            .unwrap()
            .device;

        let attachment = hal::pass::Attachment {
            format: Some(swapchain_state.format),
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

        let render_pass = unsafe {
            device.create_render_pass(&[attachment], &[subpass], &[dependency])
        }.expect("Can't create render pass");
    
        Self {
            render_pass: Some(render_pass),
            device_state: device_state.clone(),
        }
    }
}

struct BufferState<B: hal::Backend> {
    buffer: Option<B::Buffer>,
    buffer_memory: Option<B::Memory>,
    size: u64,
    device_state: Arc<RwLock<DeviceState<B>>>,
}

impl<B: hal::Backend> BufferState<B> {
    fn get_buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().unwrap()
    } 
    
    unsafe fn new<T>(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        data_source: &[T],
        usage: hal::buffer::Usage,
        memory_types: &[MemoryType]
    ) -> Self
        where T: Copy, 
              T: std::fmt::Debug
    {
        let memory: B::Memory;
        let mut buffer: B::Buffer;
        let size: u64;

        let stride = std::mem::size_of::<T>() as u64;
        let upload_size = data_source.len() as u64 * stride;

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
                let mut data_target = device
                    .acquire_mapping_writer::<T>(&memory, 0..size)
                    .unwrap();
                data_target[0..data_source.len()].copy_from_slice(data_source);
                device.release_mapping_writer(data_target).unwrap();
            }
        }

        BufferState {
            buffer_memory: Some(memory),
            buffer: Some(buffer),
            device_state: device_state.clone(),
            size,
        }
    }
}

struct PipelineState<B: hal::Backend> {
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> PipelineState<B> {
    unsafe fn new(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        render_pass: &B::RenderPass,
        descriptor_set_layouts: Vec<&B::DescriptorSetLayout>,
    ) -> Self {
        let device = &device_state
            .read()
            .unwrap()
            .device;

        let pipeline_layout = device
            .create_pipeline_layout(
                descriptor_set_layouts,
                &[(hal::pso::ShaderStageFlags::VERTEX, 0..8)],
            )
            .expect("Can't create pipeline layout");

        let vs_module = {
            let glsl = fs::read_to_string("src/data/shaders/standard.vert").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();

            device.create_shader_module(&spirv).unwrap()
        };

        let fs_module = {
            let glsl = fs::read_to_string("src/data/shaders/standard.frag").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();

            device.create_shader_module(&spirv).unwrap()
        };

        let pipeline = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint {
                    entry: "main",
                    module: &vs_module,
                    specialization: hal::pso::Specialization::default(),
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
                &pipeline_layout,
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
                    format: Format::Rgb32Sfloat,
                    offset: 0,
                },
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: hal::pso::Element {
                    format: Format::Rgb32Sfloat,
                    offset: 12,
                },
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 2,
                binding: 0,
                element: hal::pso::Element {
                    format: Format::Rg32Sfloat,
                    offset: 24,
                },
            });
            
            device.create_graphics_pipeline(&pipeline_desc, None)
        };

        // clean up shader resources
        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);

        Self {
            pipeline: Some(pipeline.unwrap()),
            pipeline_layout: Some(pipeline_layout),
            device_state: device_state.clone()
        }
    }
}

struct SwapchainState<B: hal::Backend> {
    swapchain: Option<B::Swapchain>,
    backbuffer: Option<Vec<B::Image>>,
    format: Format,
    extent: hal::image::Extent,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> SwapchainState<B> {
    fn new(backend_state: &mut BackendState<B>, device_state: &Arc<RwLock<DeviceState<B>>>) -> Self {
        let (caps, formats, _present_modes) = backend_state
            .surface
            .compatibility(&device_state.read().unwrap().physical_device);
        
        println!("formats: {:?}", formats);
        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = SwapchainConfig::from_caps(&caps, format, DIMS);
        println!("swap_config: {:?}", swap_config);
        let extent = swap_config.extent.to_extent();

        let (swapchain, backbuffer) = unsafe { 
            device_state
                .write()
                .unwrap()
                .device
                .create_swapchain(&mut backend_state.surface, swap_config, None) 
        }.expect("Can't create swapchain");

        Self {
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            format,
            extent,
            device_state: device_state.clone(),
        }
    }
}

struct FramebufferState<B: hal::Backend> {
    framebuffers: Option<Vec<B::Framebuffer>>,
    framebuffer_fences: Option<Vec<B::Fence>>,
    command_pools: Option<Vec<hal::CommandPool<B, hal::Graphics>>>,
    frame_images: Option<Vec<(B::Image, B::ImageView)>>,
    acquire_semaphores: Option<Vec<B::Semaphore>>,
    present_semaphores: Option<Vec<B::Semaphore>>,
    last_ref: usize,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> FramebufferState<B> {
    unsafe fn new(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        swapchain_state: &mut SwapchainState<B>,
        render_pass_state: &RenderPassState<B>
    ) -> Self
    {
        let (frame_images, framebuffers) = {
            let extent = hal::image::Extent {
                width: swapchain_state.extent.width as _,
                height: swapchain_state.extent.height as _,
                depth: 1,
            };

            let pairs = swapchain_state
                .backbuffer
                .take()
                .unwrap()
                .into_iter()
                .map(|image| {
                    let rtv = device_state
                        .read()
                        .unwrap()
                        .device
                        .create_image_view(
                            &image,
                            hal::image::ViewKind::D2,
                            swapchain_state.format,
                            Swizzle::NO,
                            COLOR_RANGE.clone(),
                        )
                        .unwrap();
                    (image, rtv)
                })
                .collect::<Vec<_>>();

            let fbos = pairs
                .iter()
                .map(|&(_, ref rtv)| {
                    device_state
                        .read()
                        .unwrap()
                        .device
                        .create_framebuffer(
                            render_pass_state.render_pass.as_ref().unwrap(),
                            Some(rtv),
                            extent,
                        )
                        .unwrap()
                })
                .collect();

            (pairs, fbos)
        };

        let iter_count = if frame_images.len() != 0 {
            frame_images.len()
        } else {
            1 // GL can have zero
        };

        let mut fences: Vec<B::Fence> = vec![];
        let mut command_pools: Vec<hal::CommandPool<B, hal::Graphics>> = vec![];
        let mut acquire_semaphores: Vec<B::Semaphore> = vec![];
        let mut present_semaphores: Vec<B::Semaphore> = vec![];

        for _ in 0..iter_count {
            fences.push(device_state.read().unwrap().device.create_fence(true).unwrap());
            command_pools.push(
                device_state
                    .read()
                    .unwrap()
                    .device
                    .create_command_pool_typed(
                        &device_state.read().unwrap().queue_group,
                        hal::pool::CommandPoolCreateFlags::empty(),
                    )
                    .expect("Can't create command pool"),
            );

            acquire_semaphores.push(device_state.read().unwrap().device.create_semaphore().unwrap());
            present_semaphores.push(device_state.read().unwrap().device.create_semaphore().unwrap());
        }

        FramebufferState {
            frame_images: Some(frame_images),
            framebuffers: Some(framebuffers),
            framebuffer_fences: Some(fences),
            command_pools: Some(command_pools),
            present_semaphores: Some(present_semaphores),
            acquire_semaphores: Some(acquire_semaphores),
            last_ref: 0,
            device_state: device_state.clone(),
        }
    }

    fn next_acq_pre_pair_index(&mut self) -> usize {
        if self.last_ref >= self.acquire_semaphores.as_ref().unwrap().len() {
            self.last_ref = 0
        }

        let ret = self.last_ref;
        self.last_ref += 1;
        ret
    }

    fn get_frame_data(
        &mut self,
        frame_id: Option<usize>,
        sem_index: Option<usize>,
    ) -> (
        Option<(
            &mut B::Fence,
            &mut B::Framebuffer,
            &mut hal::CommandPool<B, ::hal::Graphics>,
        )>,
        Option<(&mut B::Semaphore, &mut B::Semaphore)>,
    ) {
        (
            if let Some(fid) = frame_id {
                Some((
                    &mut self.framebuffer_fences.as_mut().unwrap()[fid],
                    &mut self.framebuffers.as_mut().unwrap()[fid],
                    &mut self.command_pools.as_mut().unwrap()[fid],
                ))
            } else {
                None
            },
            if let Some(sid) = sem_index {
                Some((
                    &mut self.acquire_semaphores.as_mut().unwrap()[sid],
                    &mut self.present_semaphores.as_mut().unwrap()[sid],
                ))
            } else {
                None
            },
        )
    }
}
    
struct ImageState<B: hal::Backend> {
    desc_set: DescSet<B>,
    sampler: Option<B::Sampler>,
    image: Option<B::Image>,
    image_view: Option<B::ImageView>,
    image_memory: Option<B::Memory>,
    transferred_image_fence: Option<B::Fence>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> ImageState<B> {
    unsafe fn new(
        desc_set: DescSet<B>,
        device_state: &Arc<RwLock<DeviceState<B>>>,
        adapter_state: &AdapterState<B>,
        usage: hal::buffer::Usage,
        command_pool: &mut CommandPool<B, Graphics>
    ) -> Self {
        let img_data = include_bytes!("data/textures/demo.jpg");
        let img = load_image(Cursor::new(&img_data[..]), image::JPEG)
            .unwrap()
            .to_rgba();

        let (width, height) = img.dimensions();
        let kind = hal::image::Kind::D2(width as hal::image::Size, height as hal::image::Size, 1, 1);
        let row_alignment_mask = adapter_state.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_stride = 4_usize;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = (height * row_pitch) as u64;

        let mut image_upload_buffer = device_state.read().unwrap().device.create_buffer(upload_size, hal::buffer::Usage::TRANSFER_SRC).unwrap();
        let image_mem_reqs = device_state.read().unwrap().device.get_buffer_requirements(&image_upload_buffer);

        let upload_type = adapter_state
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                image_mem_reqs.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let image_upload_memory = device_state.read().unwrap().device.allocate_memory(upload_type, image_mem_reqs.size).unwrap();

        device_state.read().unwrap().device.bind_buffer_memory(&image_upload_memory, 0, &mut image_upload_buffer).unwrap();

        let mut data = device_state.read().unwrap().device
            .acquire_mapping_writer::<u8>(&image_upload_memory, 0..image_mem_reqs.size)
            .unwrap();

        for y in 0..height as usize {
            let row = &(*img)[y * (width as usize) * image_stride..(y+1) * (width as usize) * image_stride];
            let dest_base = y * row_pitch as usize;
            data[dest_base..dest_base + row.len()].copy_from_slice(row);
        }

        device_state.read().unwrap().device.release_mapping_writer(data).unwrap();

        let mut image = device_state.read().unwrap().device
            .create_image(
                kind,
                1,
                Rgba8Srgb::SELF,
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

        device_state.write().unwrap().queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&mut transferred_image_fence));

        device_state.read().unwrap().device.destroy_buffer(image_upload_buffer);
        device_state.read().unwrap().device.free_memory(image_upload_memory);

        let image_view = device_state.read().unwrap().device
            .create_image_view(
                &image,
                hal::image::ViewKind::D2,
                Rgba8Srgb::SELF,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            ).unwrap();

        let sampler = device_state.read().unwrap().device
            .create_sampler(hal::image::SamplerInfo::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp))
            .expect("Can't create sampler");

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
        let device = &self.desc_set.desc_set_layout.device_state.read().unwrap().device;
        unsafe {
            device
                .wait_for_fence(&self.transferred_image_fence.as_ref().unwrap(), !0)
                .unwrap();
        }
    }

    pub fn get_layout(&self) -> &B::DescriptorSetLayout {
        self.desc_set.desc_set_layout.layout.as_ref().unwrap()
    }
}

struct DescSet<B: hal::Backend> {
    descriptor_set: B::DescriptorSet,
    desc_set_layout: DescSetLayout<B>,
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

    fn create_set(self, descriptor_pool: &mut B::DescriptorPool) -> DescSet<B> {
        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(&self.layout.as_ref().unwrap())
        }.unwrap();
       
        DescSet {
            descriptor_set,
            desc_set_layout: self
        }
    }
}

pub struct Renderer<B: hal::Backend> {
    image_desc_pool: Option<B::DescriptorPool>,
    viewport: Viewport,

    backend_state: BackendState<B>,
    device_state: Arc<RwLock<DeviceState<B>>>,
    swapchain_state: SwapchainState<B>,
    window_state: WindowState,
    render_pass_state: RenderPassState<B>,
    pipeline_state: PipelineState<B>,
    framebuffer_state: FramebufferState<B>,
    
    image_state: ImageState<B>,
    vertex_buffer_state: BufferState<B>,

    recreate_swapchain: bool,
    resize_dims: Extent2D,
}

impl<B: hal::Backend> Renderer<B> {
    pub unsafe fn new(mut backend_state: BackendState<B>, window_state: WindowState) -> Self {
        let device_state = Arc::new(
            RwLock::new(
                DeviceState::new(
                    backend_state.adapter_state.adapter.take().unwrap(),
                    &backend_state.surface
                )
            )
        );

        let mut image_desc_pool =
            device_state
                .read()
                .unwrap()
                .device
                .create_descriptor_pool(
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
                    ],
                    hal::pso::DescriptorPoolCreateFlags::empty()
                )
                .expect("Can't create descriptor pool");

        let image_desc_set_layout = DescSetLayout::new(
            &device_state,
            vec![
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
            ]
        );

        let image_desc_set = image_desc_set_layout.create_set(&mut image_desc_pool);

        let mut staging_pool = device_state
            .read()
            .unwrap()
            .device
            .create_command_pool_typed(
                &device_state
                    .read()
                    .unwrap()
                    .queue_group,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
            .expect("Can't create staging command pool");

        let image_state = ImageState::new(
            image_desc_set,
            &device_state,
            &backend_state.adapter_state,
            hal::buffer::Usage::TRANSFER_SRC,
            &mut staging_pool
        );

        device_state
            .read()
            .unwrap()
            .device
            .destroy_command_pool(staging_pool.into_raw());

        let vertex_buffer_state = BufferState::new(
            &device_state,
            &QUAD,
            hal::buffer::Usage::VERTEX,
            &backend_state.adapter_state.memory_types,
        );
       
        let mut swapchain_state = SwapchainState::new(&mut backend_state, &device_state);
        let render_pass_state = RenderPassState::new(&device_state, &swapchain_state);
        let framebuffer_state = FramebufferState::new(&device_state, &mut swapchain_state, &render_pass_state);
        let pipeline_state = PipelineState::new(
            &device_state,
            render_pass_state.render_pass.as_ref().unwrap(),
            vec![image_state.get_layout()]
        );
        let viewport = Self::create_viewport(&swapchain_state);

        let resize_dims = Extent2D {
            width: 0,
            height: 0,
        };
        
        Self {
            image_desc_pool: Some(image_desc_pool),
            viewport,

            backend_state,
            device_state,
            swapchain_state,
            window_state,
            render_pass_state,
            pipeline_state,
            framebuffer_state,

            image_state,
            vertex_buffer_state,

            recreate_swapchain: false,
            resize_dims,
        }
    }

    fn create_viewport(swapchain_state: &SwapchainState<B>) -> hal::pso::Viewport {
        hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: swapchain_state.extent.width as _,
                h: swapchain_state.extent.height as _,
            },
            depth: 0.0..1.0,
        }
    }

    pub unsafe fn draw_frame(&mut self, events_loop: &mut Arc<RwLock<EventHandler>>) {
        self.window_state.events_loop.poll_events(|winit_event| {
            if let winit::Event::WindowEvent { event, ..  } = winit_event {
                match event {
                    winit::WindowEvent::KeyboardInput {
                        input:winit::KeyboardInput {
                                virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    }
                    | winit::WindowEvent::CloseRequested => panic!("matthew's bad way of handling exit"),
                    // winit::WindowEvent::Resized(dims) => {
                    //     println!("resized to: {:?}", dims);

                    //     #[cfg(feature = "gl")]
                    //     self.surface
                    //         .get_window()
                    //         .resize(dims.to_physical(self.surface.get_window().get_hidpi_factor()));

                    //     self.recreate_swapchain = true;
                    //     self.resize_dims.width = dims.width as u32;
                    //     self.resize_dims.height = dims.height as u32;
                    // }
                    _ => (),
                }
            }
        });

        // TODO -> figure out event handling
        // events_loop.lock().unwrap().poll_events(|event| { if let winit::Event::WindowEvent { event, .. } = event {
        //         match event {
        //             winit::WindowEvent::KeyboardInput {
        //                 input:
        //                     winit::KeyboardInput {
        //                         virtual_keycode: Some(winit::VirtualKeyCode::Escape),
        //                         ..
        //                     },
        //                 ..
        //             }
        //                 | winit::WindowEvent::CloseRequested => panic!("matthew's bad way of handling exit"),
        //             winit::WindowEvent::Resized(dims) => {
        //                 println!("resized to: {:?}", dims);

        //                 #[cfg(feature = "gl")]
        //                 self.surface
        //                     .get_window()
        //                     .resize(dims.to_physical(self.surface.get_window().get_hidpi_factor()));

        //                 self.recreate_swapchain = true;
        //                 self.resize_dims.width = dims.width as u32;
        //                 self.resize_dims.height = dims.height as u32;
        //             }
        //             _ => (),
        //         }
        //     }
        // });

        if self.recreate_swapchain {
            self.recreate_swapchain();
            self.recreate_swapchain = false; 
        }

        let sem_index = self.framebuffer_state.next_acq_pre_pair_index();

        let frame: hal::SwapImageIndex = {
            let (acquire_semaphore, _) = self
                .framebuffer_state
                .get_frame_data(None, Some(sem_index))
                .1
                .unwrap();

            match self
                .swapchain_state
                .swapchain
                .as_mut()
                .unwrap()
                .acquire_image(!0, Some(acquire_semaphore), None)
            {
                Ok((i, _)) => i,
                Err(e) => {
					println!("we gots an error on AQUIREIMAGE: {:?}", e);
                    self.recreate_swapchain = true;
                    return;
                }
            }
        };

        let (fid, sid) = self.framebuffer_state
            .get_frame_data(Some(frame as usize), Some(sem_index));

        let (framebuffer_fence, framebuffer, command_pool) = fid.unwrap();
        let (image_acquired, image_present) = sid.unwrap();

        self.device_state
            .read()
            .unwrap()
            .device
            .wait_for_fence(framebuffer_fence, !0)
            .unwrap();

        self.device_state
            .read()
            .unwrap()
            .device
            .reset_fence(framebuffer_fence)
            .unwrap();

        command_pool.reset();

        // Rendering
        let mut cmd_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
        cmd_buffer.begin();

        cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
        cmd_buffer.set_scissors(0, &[self.viewport.rect]);
        cmd_buffer.bind_graphics_pipeline(&self.pipeline_state.pipeline.as_ref().unwrap());
        cmd_buffer.bind_vertex_buffers(0, Some((self.vertex_buffer_state.get_buffer(), 0)));
        cmd_buffer.bind_graphics_descriptor_sets(
            &self.pipeline_state.pipeline_layout.as_ref().unwrap(),
            0,
            vec![
                &self.image_state.desc_set.descriptor_set,
            ],
            &[],
        ); //TODO

        {
            let mut encoder = cmd_buffer.begin_render_pass_inline(
                self.render_pass_state.render_pass.as_ref().unwrap(),
                framebuffer,
                self.viewport.rect,
                &[hal::command::ClearValue::Color(hal::command::ClearColor::Float([
                    0.6, 0.2, 0.0, 1.0,
                ]))],
            );
            encoder.draw(0..(QUAD.len() as u32), 0..1);
        }
        cmd_buffer.finish();

        let submission = Submission {
            command_buffers: std::iter::once(&cmd_buffer),
            wait_semaphores: std::iter::once((&*image_acquired, PipelineStage::BOTTOM_OF_PIPE)),
            signal_semaphores: std::iter::once(&*image_present),
        };

        self.device_state
            .write()
            .unwrap()
            .queue_group.queues[0]
            .submit(submission, Some(framebuffer_fence));

        // present frame
        if let Err(e) = self
            .swapchain_state
            .swapchain
            .as_mut()
            .unwrap()
            .present(
                &mut self.device_state.write().unwrap().queue_group.queues[0],
                frame,
                Some(&*image_present),
            )
        {
			println!("well boys we hit an error: {:?}", e);
            self.recreate_swapchain = true;
        }
    }

    unsafe fn recreate_swapchain(&mut self) {
        self.device_state.read().unwrap().device.wait_idle().unwrap();

        self.swapchain_state = SwapchainState::new(&mut self.backend_state, &self.device_state);

        self.render_pass_state = RenderPassState::new(&self.device_state, &self.swapchain_state);

        self.framebuffer_state = FramebufferState::new(
            &self.device_state,
            &mut self.swapchain_state,
            &self.render_pass_state,
        );

        self.pipeline_state = PipelineState::new(
            &self.device_state,
            self.render_pass_state.render_pass.as_ref().unwrap(),
            vec![self.image_state.get_layout()],
        );

        self.viewport = Self::create_viewport(&self.swapchain_state);
    }
}

impl<B: hal::Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        self.device_state.read().unwrap().device.wait_idle().unwrap();
        unsafe {
            self.device_state
                .read()
                .unwrap()
                .device
                .destroy_descriptor_pool(self.image_desc_pool.take().unwrap());
        }
    }
}


impl<B: hal::Backend> Drop for RenderPassState<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_render_pass(self.render_pass.take().unwrap());
        }
    }
}


impl<B: hal::Backend> Drop for BufferState<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_buffer(self.buffer.take().unwrap());
            device.free_memory(self.buffer_memory.take().unwrap());
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


impl<B: hal::Backend> Drop for ImageState<B> {
    fn drop(&mut self) {
        unsafe {
            let device = &self.desc_set.desc_set_layout.device_state.read().unwrap().device;

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

impl<B: hal::Backend> Drop for PipelineState<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_graphics_pipeline(self.pipeline.take().unwrap());
            device.destroy_pipeline_layout(self.pipeline_layout.take().unwrap());
        }
    }
}

impl<B: hal::Backend> Drop for SwapchainState<B> {
    fn drop(&mut self) {
        unsafe {
            self.device_state
                .read()
                .unwrap()
                .device
                .destroy_swapchain(self.swapchain.take().unwrap());
        }
    }
}

impl<B: hal::Backend> Drop for FramebufferState<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;

        unsafe {
            for fence in self.framebuffer_fences.take().unwrap() {
                device.wait_for_fence(&fence, !0).unwrap();
                device.destroy_fence(fence);
            }

            for command_pool in self.command_pools.take().unwrap() {
                device.destroy_command_pool(command_pool.into_raw());
            }

            for acquire_semaphore in self.acquire_semaphores.take().unwrap() {
                device.destroy_semaphore(acquire_semaphore);
            }

            for present_semaphore in self.present_semaphores.take().unwrap() {
                device.destroy_semaphore(present_semaphore);
            }

            for framebuffer in self.framebuffers.take().unwrap() {
                device.destroy_framebuffer(framebuffer);
            }

            for (_, rtv) in self.frame_images.take().unwrap() {
                device.destroy_image_view(rtv);
            }
        }
    }
}

