use std::sync::{Arc, RwLock};
use std::collections::{HashMap, BTreeMap};

use crate::components::{mesh::Mesh, texture::Texture, transform::Transform};
use crate::primitives::{drawable::Drawable, vertex::Vertex};
use crate::primitives::uniform_buffer_object::{CameraUniformBufferObject, ObjectUniformBufferObject};
use crate::renderer::allocator::{COLOR_RANGE, Allocator, GfxAllocator};
use crate::renderer::types::{Image, Uniform, Buffer, DescSetLayout};
use crate::renderer::render_key::RenderKey;
use crate::renderer::core::RendererCore;
use crate::utils::data_path;

use cgmath::{Vector3, Matrix4};
use cgmath::Angle;

use itertools::Itertools;

use hal::pool::CommandPool;
use hal::command::CommandBuffer;
use hal::pso::Viewport;
use hal::device::Device;
use hal::queue::CommandQueue;

pub(crate) trait Drawer<B: hal::Backend> {
    fn draw(&mut self, image_index: usize) -> &B::Semaphore;
    fn update_drawables(&mut self, drawables: Vec<Drawable>) -> Result<(), String>;
    fn update_uniforms(&mut self, uniforms: Vec<ObjectUniformBufferObject>) -> Result<(), String>;
    fn update_camera(&mut self, transform: Transform) -> Result<(), String>;
}

pub(crate) struct GfxDrawer<B: hal::Backend, A: Allocator<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: A,

    framebuffers: Framebuffers<B>,
    render_pass: RenderPass<B>,
    pipeline: Pipeline<B>,
    viewport: Viewport,

    image_desc_set_layout: DescSetLayout<B>,
    images: HashMap<RenderKey, Image<B>>,

    vertex_buffer: Option<Buffer<B>>,
    index_buffer: Option<Buffer<B>>,

    camera_uniform: Uniform<B>,
    object_uniform: Uniform<B>,

    last_drawables: Option<Vec<Drawable>>,
}

impl <B: hal::Backend> GfxDrawer<B, GfxAllocator<B>> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>, mut allocator: GfxAllocator<B>, viewport: Viewport) -> Self {
        let render_pass = RenderPass::new(
            core,
            swapchain_format,
        );

        let framebuffers = unsafe {
            Framebuffers::new(
                &core,
                hal::image::Extent {
                    width: viewport.rect.w as u32,
                    height: viewport.rect.h as u32,
                    depth: viewport.depth.end as u32,
                },
                images,
                image_format,
                &render_pass,
                depth_image_stuff,
            )
        };

        let camera_uniform = Self::init_uniform(
            &mut allocator,
            &vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Buffer {
                    ty: hal::pso::BufferDescriptorType::Uniform,
                    format: hal::pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    }
                },
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }],
            &[CameraUniformBufferObject::default()]
        );

        let object_uniform = Self::init_uniform(
            &mut allocator,
            &vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Buffer {
                    ty: hal::pso::BufferDescriptorType::Uniform,
                    format: hal::pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: true,
                    }
                },
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }],
            &[ObjectUniformBufferObject::default()],
        );

        let image_desc_set_layout = allocator.alloc_desc_set_layout(
            &vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Image {
                    ty: hal::pso::ImageDescriptorType::Sampled {
                        with_sampler: true,
                    }
                },
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            }]);

        let pipeline = unsafe {
            Pipeline::new(
                core,
                render_pass.render_pass.as_ref().unwrap(),
                vec![
                    camera_uniform.desc.as_ref().unwrap().desc_set_layout.layout.as_ref().unwrap(),
                    object_uniform.desc.as_ref().unwrap().desc_set_layout.layout.as_ref().unwrap(),
                    image_desc_set_layout.layout.as_ref().unwrap()
                ],
                "shaders/standard.vert",
                "shaders/standard.frag",
            )
        };

        Self {
            core: Arc::clone(core),
            allocator,
            framebuffers,
            render_pass,
            pipeline,
            viewport,
            image_desc_set_layout,
            images: HashMap::new(),
            vertex_buffer: None,
            index_buffer: None,
            camera_uniform,
            object_uniform,
            last_drawables: None
        }
    }

    // TODO -> is there a way to streamline uniform allocation so that it encapsulates DescSetLayouts and DescSets?
    fn init_uniform<T>(allocator: &mut GfxAllocator<B>, bindings: &[hal::pso::DescriptorSetLayoutBinding], data: &[T])-> Uniform<B>
        where T: Copy,
              T: std::fmt::Debug
    {
        let desc_set_layout = allocator.alloc_desc_set_layout(bindings);
        let desc_set = allocator.alloc_desc_set(&desc_set_layout);
        allocator.alloc_uniform(data, desc_set, 0)
    }

    unsafe fn generate_vertex_and_index_buffers(&mut self, meshes: Vec<&Mesh>) {
        let vertices = meshes
            .iter()
            .flat_map(|m| {
                m.vertices.iter().map(|v| *v)
            })
            .collect::<Vec<Vertex>>();

        let mut current_index = 0;
        let indices = meshes
            .iter()
            .flat_map(|m| {
                let indices = m.indices
                    .iter()
                    .map(move |val| current_index + val);

                current_index += m.vertices.len() as u32;

                return indices;
            })
            .collect::<Vec<u32>>();

        let vertex_alignment = self.core.read().unwrap().backend.adapter.limits.min_vertex_input_binding_stride_alignment;
        let vertex_buffer = self.allocator.alloc_buffer(
            &vertices,
            vertex_alignment,
            65536,
            hal::buffer::Usage::VERTEX,
        );

        let index_buffer = self.allocator.alloc_buffer(
            &indices,
            1,
            65536,
            hal::buffer::Usage::INDEX,
        );

        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
    }


    unsafe fn generate_images(&mut self, textures: Vec<&Texture>) {
        let new_textures: Vec<&Texture> = textures
            .into_iter()
            .filter(|t| !self.images.contains_key(&RenderKey::from(*t)))
            .unique()
            .collect();

        if new_textures.is_empty() {
            return;
        }

        for texture in new_textures.into_iter() {
            let image = self.allocator.alloc_image(
                hal::buffer::Usage::TRANSFER_SRC,
                &texture.path,
                &hal::image::SamplerDesc::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp),
            );
            self.images.insert(RenderKey::from(texture), image);
        }
    }

    // TODO -> this shouldn't be in drawer
    // Pitch must be in the range of [-90 ... 90] degrees and
    // yaw must be in the range of [0 ... 360] degrees.
    // Pitch and yaw variables must be expressed in radians.
    pub fn update_camera_uniform_buffer_object(&self, dimensions: [f32;2], camera_transform: &Transform) -> CameraUniformBufferObject {
        let position = camera_transform.position;
        let rotation = cgmath::Euler::from(camera_transform.rotation);

        let view = fps_view_matrix(position, rotation.y, rotation.x);

        let mut proj = cgmath::perspective(
            cgmath::Deg(45.0),
            dimensions[0] / dimensions[1],
            0.1,
            1000.0
        );

        proj.y.y *= -1.0;

        CameraUniformBufferObject::new(view, proj)
    }

    unsafe fn generate_cmd_buffers(&mut self , meshes_by_texture: BTreeMap<Option<Texture>, Vec<&Mesh>>) {
        let framebuffers = self.framebuffers
            .framebuffers
            .as_ref()
            .unwrap();

        let command_pools = self.framebuffers
            .command_pools
            .as_mut()
            .unwrap();

        let num_buffers = framebuffers.len();

        // TODO -> assert all sizes are same and all options are "Some"

        let mut command_buffers: Vec<B::CommandBuffer> = Vec::new();

        for current_buffer_index in 0..num_buffers {
            let framebuffer = &framebuffers[current_buffer_index];
            let command_pool = &mut command_pools[current_buffer_index];

            // Rendering
            let mut cmd_buffer = command_pool.allocate_one(hal::command::Level::Primary);
            cmd_buffer.begin_primary(hal::command::CommandBufferFlags::SIMULTANEOUS_USE);

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);

            cmd_buffer.bind_graphics_pipeline(&self.pipeline.pipeline.as_ref().unwrap());
            cmd_buffer.bind_vertex_buffers(0, Some((self.vertex_buffer.as_ref().unwrap().get_buffer(), 0)));
            cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                buffer: self.index_buffer.as_ref().unwrap().get_buffer(),
                offset: 0,
                index_type: hal::IndexType::U32
            });

            cmd_buffer.begin_render_pass(
                self.render_pass.render_pass.as_ref().unwrap(),
                &framebuffer,
                self.viewport.rect,
                &[
                    hal::command::ClearValue { color: hal::command::ClearColor { float32: [0.7, 0.2, 0.0, 1.0] } },
                    hal::command::ClearValue { depth_stencil: hal::command::ClearDepthStencil {depth: 1.0, stencil: 0} }
                ],
                hal::command::SubpassContents::Inline
            );

            let mut current_mesh_index = 0;

            for (maybe_texture, meshes) in meshes_by_texture.iter() {
                let texture_key = RenderKey::from(maybe_texture);
                let texture_image = self.images.get(&texture_key).unwrap();

                cmd_buffer.bind_graphics_descriptor_sets(
                    &self.pipeline.pipeline_layout.as_ref().unwrap(),
                    2,
                    vec![ &texture_image.desc_set.descriptor_set ],
                    &[],
                );

                for (i, mesh) in meshes.iter().enumerate() {
                    if !mesh.rendered {
                        continue;
                    }

                    let dynamic_offset = i as u64 * self.object_uniform.buffer.as_ref().unwrap().padded_stride;

                    cmd_buffer.bind_graphics_descriptor_sets(
                        &self.pipeline.pipeline_layout.as_ref().unwrap(),
                        0,
                        vec![
                            &self.camera_uniform.desc.as_ref().unwrap().descriptor_set,
                            &self.object_uniform.desc.as_ref().unwrap().descriptor_set,
                        ],
                        &[dynamic_offset as u32],
                    );

                    let num_indices = mesh.indices.len() as u32;
                    cmd_buffer.draw_indexed(current_mesh_index..(current_mesh_index + num_indices), 0, 0..1);
                    current_mesh_index += num_indices;
                }
            }

            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            command_buffers.push(cmd_buffer);
        }

        self.framebuffers.command_buffers = Some(command_buffers);
    }
}

impl <B: hal::Backend, A: Allocator<B>> Drop for GfxDrawer<B, A> {
    fn drop(&mut self) {
        let readable_core = self.core.read().unwrap();
        let raw_device = &mut readable_core.device.device;
        self.image_desc_set_layout.drop(raw_device);
        self.vertex_buffer.take().unwrap().drop(raw_device);
        self.index_buffer.take().unwrap().drop(raw_device);
        self.camera_uniform.drop(raw_device);
        self.object_uniform.drop(raw_device);
        for image in self.images.values_mut() {
            image.drop(raw_device);
        }
    }
}

pub fn fps_view_matrix(eye: Vector3<f32>, pitch_rad: cgmath::Rad<f32>, yaw_rad: cgmath::Rad<f32>) -> Matrix4<f32> {
    let cos_pitch = pitch_rad.cos();
    let sin_pitch = pitch_rad.sin();

    let cos_yaw = yaw_rad.cos();
    let sin_yaw = yaw_rad.sin();

    let x_axis = Vector3::new(cos_yaw, 0.0, -sin_yaw);
    let y_axis = Vector3::new(sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch);
    let z_axis = Vector3::new(sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw);

    let view_matrix = Matrix4::new(
        x_axis.x, y_axis.x, z_axis.x, 0.0,
        x_axis.y, y_axis.y, z_axis.y, 0.0,
        x_axis.z, y_axis.z, z_axis.z, 0.0,
        cgmath::dot(x_axis, eye) * -1.0, cgmath::dot(y_axis, eye) * -1.0, cgmath::dot(z_axis, eye) * -1.0, 1.0
    );

    view_matrix
}

impl <B: hal::Backend> Drawer<B> for GfxDrawer<B, GfxAllocator<B>> {
    fn draw(&mut self, image_index: usize) -> &B::Semaphore {
        unsafe {

            // TODO: THIS SHOULD BE REFACTORED
            let sem_index = self.framebuffers.next_acq_pre_pair_index();
            let (fid, sid) = self.framebuffers.get_frame_data(Some(image_index), Some(sem_index));
            let (framebuffer_fence, command_buffer) = fid.unwrap();
            let (image_acquired_semaphore, image_present_semaphore) = sid.unwrap();

            self
                .core
                .read()
                .unwrap()
                .device
                .device
                .wait_for_fence(framebuffer_fence, !0)
                .unwrap();

            self
                .core
                .read()
                .unwrap()
                .device
                .device
                .reset_fence(framebuffer_fence)
                .unwrap();

            let submission = hal::queue::Submission {
                command_buffers: std::iter::once(&*command_buffer),
                wait_semaphores: std::iter::once((&*image_acquired_semaphore, hal::pso::PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: std::iter::once(&*image_present_semaphore),
            };

            self
                .core
                .write()
                .unwrap()
                .device
                .queue_group
                .queues[0]
                .submit(submission, Some(framebuffer_fence));

            image_acquired_semaphore
        }
    }

    fn update_drawables(&mut self, drawables: Vec<Drawable>) -> Result<(), String> {
        unsafe {
            let uniforms = drawables
                .iter()
                .map(|d| d.transform.to_ubo())
                .collect::<Vec<ObjectUniformBufferObject>>();

            self.update_uniforms(uniforms);

            self.generate_images(
                drawables
                    .iter()
                    .filter(|d| d.texture.is_some())
                    .map(|d| d.texture.as_ref().unwrap())
                    .collect());

            self.generate_vertex_and_index_buffers(
                drawables
                    .iter_mut()
                    .map(|d| &d.mesh)
                    .collect::<Vec<&Mesh>>()
            );

            let meshes_by_texture = drawables
                .iter()
                .fold(BTreeMap::<Option<Texture>, Vec<&Mesh>>::new(), |mut map, drawable| {
                    let meshes_by_texture = map.entry(drawable.texture.clone()).or_insert(Vec::new());
                    meshes_by_texture.push(&drawable.mesh);
                    map
                });

            self.generate_cmd_buffers(meshes_by_texture);
            self.last_drawables = Some(drawables);

            Ok(())
        }
    }

    fn update_uniforms(&mut self, uniforms: Vec<ObjectUniformBufferObject>) -> Result<(), String> {
        unsafe {
            self
                .object_uniform
                .buffer
                .as_mut()
                .unwrap()
                .update_data(&self.core, 0, &uniforms);
        }

        Ok(())
    }

    fn update_camera(&mut self, transform: Transform) -> Result<(), String> {
        let new_ubo = self.update_camera_uniform_buffer_object(self.dims, transform);
        self
            .camera_uniform
            .buffer
            .as_mut()
            .unwrap()
            .update_data(&self.core, 0, &[new_ubo]);

        Ok(())
    }
}

struct RenderPass<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,
    render_pass: Option<B::RenderPass>,
}

impl<B: hal::Backend> RenderPass<B> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>, swapchain_format: hal::format::Format) -> Self {
        let device = &core
            .read()
            .unwrap()
            .device
            .device;

        let color_attachment = hal::pass::Attachment {
            format: Some(swapchain_format),
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
            stages: hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT..hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: hal::image::Access::empty()..(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
            flags: hal::memory::Dependencies::empty(),
        };

        let render_pass = unsafe {
            device.create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[dependency])
        }.expect("Can't create render pass");

        Self {
            core: Arc::clone(core),
            render_pass: Some(render_pass),
        }
    }
}

impl<B: hal::Backend> Drop for RenderPass<B> {
    fn drop(&mut self) {
        let device = &self.core.read().unwrap().device.device;
        unsafe {
            device.destroy_render_pass(self.render_pass.take().unwrap());
        }
    }
}

struct Pipeline<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
}

impl<B: hal::Backend> Pipeline<B> {
    unsafe fn new(
        core: &Arc<RwLock<RendererCore<B>>>,
        render_pass: &B::RenderPass,
        descriptor_set_layouts: Vec<&B::DescriptorSetLayout>,
        vertex_shader: &str,
        fragment_shader: &str
    ) -> Self {
        let device = &core
            .read()
            .unwrap()
            .device
            .device;

        let pipeline_layout = device
            .create_pipeline_layout(
                descriptor_set_layouts,
                &[(hal::pso::ShaderStageFlags::VERTEX, 0..8)],
            )
            .expect("Can't create pipeline layout");

        let load_shader = |shader_path, shader_type| {
            let glsl = std::fs::read_to_string(data_path(shader_path)).unwrap();
            let mut spirv_file = glsl_to_spirv::compile(&glsl, shader_type).unwrap();
            let spirv = hal::pso::read_spirv(&mut spirv_file).unwrap();
            device.create_shader_module(&spirv[..]).unwrap()
        };

        let vs_module = load_shader(vertex_shader, glsl_to_spirv::ShaderType::Vertex);
        let fs_module = load_shader(fragment_shader, glsl_to_spirv::ShaderType::Fragment);

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

            let subpass = hal::pass::Subpass {
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
                rate: hal::pso::VertexInputRate::Vertex,
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: hal::pso::Element {
                    format: hal::format::Format::Rgb32Sfloat,
                    offset: 0,
                },
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: hal::pso::Element {
                    format: hal::format::Format::Rgb32Sfloat,
                    offset: 12,
                },
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                location: 2,
                binding: 0,
                element: hal::pso::Element {
                    format: hal::format::Format::Rg32Sfloat,
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
            core: Arc::clone(core),
            pipeline: Some(pipeline.unwrap()),
            pipeline_layout: Some(pipeline_layout),
        }
    }
}

impl<B: hal::Backend> Drop for Pipeline<B> {
    fn drop(&mut self) {
        let device = &self.core.read().unwrap().device.device;
        unsafe {
            device.destroy_graphics_pipeline(self.pipeline.take().unwrap());
            device.destroy_pipeline_layout(self.pipeline_layout.take().unwrap());
        }
    }
}

struct Framebuffers<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,
    framebuffers: Option<Vec<B::Framebuffer>>,
    framebuffer_fences: Option<Vec<B::Fence>>,
    command_pools: Option<Vec<B::CommandPool>>,
    command_buffers: Option<Vec<B::CommandBuffer>>,
    frame_images: Option<Vec<(B::Image, B::ImageView)>>,
    acquire_semaphores: Option<Vec<B::Semaphore>>,
    present_semaphores: Option<Vec<B::Semaphore>>,
    last_ref: usize,
    depth_image_stuff: Option<(B::Image, B::Memory, B::ImageView)>,
}

impl<B: hal::Backend> Framebuffers<B> {
    unsafe fn new(
        core: &Arc<RwLock<RendererCore<B>>>,
        extent: hal::image::Extent,
        images: Vec<B::Image>,
        image_format: hal::format::Format,
        render_pass: &RenderPass<B>,
        // TODO -> Get rid of / clean up
        depth_image_stuff: (B::Image, B::Memory, B::ImageView)
    ) -> Self
    {
        // TODO -> create depth image for each frame

        let (frame_images, framebuffers) = {
            let pairs = images
                .into_iter()
                .map(|image| {
                    let image_view = core
                        .read()
                        .unwrap()
                        .device
                        .device
                        .create_image_view(
                            &image,
                            hal::image::ViewKind::D2,
                            image_format,
                            hal::format::Swizzle::NO,
                            COLOR_RANGE.clone(),
                        )
                        .unwrap();

                    (image, image_view)
                })
                .collect::<Vec<_>>();

            let fbos = pairs
                .iter()
                .map(|&(_, ref image_view)| {
                    core
                        .read()
                        .unwrap()
                        .device
                        .device
                        .create_framebuffer(
                            render_pass.render_pass.as_ref().unwrap(),
                            vec![image_view, &depth_image_stuff.2],
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
        let mut command_pools: Vec<B::CommandPool> = vec![];
        let mut acquire_semaphores: Vec<B::Semaphore> = vec![];
        let mut present_semaphores: Vec<B::Semaphore> = vec![];

        for _ in 0..iter_count {
            fences.push(core.read().unwrap().device.device.create_fence(true).unwrap());
            command_pools.push(
                core
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
                    .expect("Can't create command pool"),
            );

            acquire_semaphores.push(core.read().unwrap().device.device.create_semaphore().unwrap());
            present_semaphores.push(core.read().unwrap().device.device.create_semaphore().unwrap());
        }

        Self {
            core: Arc::clone(core),
            frame_images: Some(frame_images),
            framebuffers: Some(framebuffers),
            framebuffer_fences: Some(fences),
            command_pools: Some(command_pools),
            command_buffers: None,
            present_semaphores: Some(present_semaphores),
            acquire_semaphores: Some(acquire_semaphores),
            last_ref: 0,
            depth_image_stuff: Some(depth_image_stuff),
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

    // struct FrameData {
    //     framebuffer_fence: B::Fence,
    //     command_buffer: B::CommandBuffer,
    //     image_acquired_semaphore: B::Semaphore,
    //     image_present_sempahore: B::Semaphore,
    // }
    fn get_frame_data(
        &mut self,
        frame_id: Option<usize>,
        sem_index: Option<usize>,
    ) -> (
        Option<(
            &mut B::Fence,
            &mut B::CommandBuffer
        )>,
        Option<(&mut B::Semaphore, &mut B::Semaphore)>,
    ) {
        (
            if let Some(fid) = frame_id {
                Some((
                    &mut self.framebuffer_fences.as_mut().unwrap()[fid],
                    &mut self.command_buffers.as_mut().unwrap()[fid]
                ))
            } else {
                None
            },
            if let Some(sid) = sem_index {
                Some((
                    &mut self.acquire_semaphores.as_mut().unwrap()[sid],
                    &mut self.present_semaphores.as_mut().unwrap()[sid]
                ))
            } else {
                None
            },
        )
    }
}

impl<B: hal::Backend> Drop for Framebuffers<B> {
    fn drop(&mut self) {
        let device = &self.core.read().unwrap().device.device;

        unsafe {
            for fence in self.framebuffer_fences.take().unwrap() {
                device.wait_for_fence(&fence, !0).unwrap();
                device.destroy_fence(fence);
            }

            for command_pool in self.command_pools.take().unwrap() {
                device.destroy_command_pool(command_pool);
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

            let depth_image_stuff = self.depth_image_stuff.take().unwrap();
            device.destroy_image_view(depth_image_stuff.2);
            device.destroy_image(depth_image_stuff.0);
            device.free_memory(depth_image_stuff.1);
        }
    }
}
