use allocator::{Image};

trait Drawer {
    fn draw(image: &mut Image);
}

struct GfxDrawer {
    render_pass_state: RenderPassState<B>,
    pipeline_state: PipelineState<B>,

    image_desc_set_layout: Option<Arc<RwLock<DescSetLayout<B>>>>,
    image_states: HashMap<RenderKey, Image<B>>,

    vertex_buffer_state: Option<Buffer<B>>,
    index_buffer_state: Option<Buffer<B>>,

    camera_uniform: Uniform<B>,
    object_uniform: Uniform<B>,

    last_drawables: Option<Vec<Drawable>>,
}

impl GfxDrawer {
    pub fn new() -> Self {

    }
}

impl Drawer for GfxDrawer {
    fn draw(image: &mut _) {
        unimplemented!()
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

        let color_attachment = hal::pass::Attachment {
            format: Some(swapchain_state.format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined..hal::image::Layout::Present,
        };

        let depth_format = hal::format::Format::D32SfloatS8Uint;
        let depth_attachment = hal::pass::Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::DontCare,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined..hal::image::Layout::DepthStencilAttachmentOptimal,
        };

        let subpass = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = hal::pass::SubpassDependency {
            passes: hal::pass::SubpassRef::External..hal::pass::SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: hal::image::Access::empty()..(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
            flags: hal::memory::Dependencies::empty(),
        };

        let render_pass = unsafe {
            device.create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[dependency])
        }.expect("Can't create render pass");

        Self {
            render_pass: Some(render_pass),
            device_state: Arc::clone(device_state),
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
        vertex_shader: &str,
        fragment_shader: &str
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
            let glsl = fs::read_to_string(data_path(vertex_shader)).unwrap();
            let mut spirv_file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex).unwrap();
            let spirv = hal::pso::read_spirv(&mut spirv_file).unwrap();
            device.create_shader_module(&spirv[..]).unwrap()
        };

        let fs_module = {
            let glsl = fs::read_to_string(data_path(fragment_shader)).unwrap();
            let mut spirv_file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment).unwrap();
            let spirv = hal::pso::read_spirv(&mut spirv_file).unwrap();
            device.create_shader_module(&spirv[..]).unwrap()
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
                hal::pso::Primitive::TriangleList,
                hal::pso::Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::ALPHA),
            });

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

            pipeline_desc.depth_stencil = hal::pso::DepthStencilDesc {
                depth: Some(hal::pso::DepthTest {
                    fun: hal::pso::Comparison::Less,
                    write: true
                }),
                depth_bounds: false,
                stencil: None,
            };

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

impl<B: hal::Backend> Drop for PipelineState<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_graphics_pipeline(self.pipeline.take().unwrap());
            device.destroy_pipeline_layout(self.pipeline_layout.take().unwrap());
        }
    }
}
