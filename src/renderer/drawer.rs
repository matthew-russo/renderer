use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use crate::renderer::allocator::{Image, Uniform};
use crate::primitives::drawable::Drawable;
use crate::renderer::render_key::RenderKey;
use crate::primitives::uniform_buffer_object::CameraUniformBufferObject;
use crate::components::transform::Transform;

trait Drawer<B: hal::Backend> {
    fn draw(image: &mut Image<B>);
}

struct GfxDrawer<B: hal::Backend> {
    framebuffers: Framebuffers<B>,
    render_pass: RenderPass<B>,
    pipeline: Pipeline<B>,

    image_desc_set_layout: Option<Arc<RwLock<DescSetLayout<B>>>>,
    image_states: HashMap<RenderKey, Image<B>>,

    vertex_buffer_state: Option<Buffer<B>>,
    index_buffer_state: Option<Buffer<B>>,

    camera_uniform: Uniform<B>,
    object_uniform: Uniform<B>,

    last_drawables: Option<Vec<Drawable>>,
}

impl GfxDrawer {
    pub fn new(device: &Arc<RwLock<Device<B>>>) -> Self {
        let render_pass = RenderPass::new(
            &device,
            &swapchain,
        );

        let pipeline_state = Pipeline::new(
            &device,
            render_pass.handle.as_ref().unwrap(),
            vec![
                camera_uniform.desc.as_ref().unwrap().desc_set_layout.read().unwrap().layout.as_ref().unwrap(),
                object_uniform.desc.as_ref().unwrap().desc_set_layout.read().unwrap().layout.as_ref().unwrap(),
                image_desc_set_layout.read().unwrap().layout.as_ref().unwrap()
            ],
            "shaders/standard.vert",
            "shaders/standard.frag",
        );

        Self {

        }
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

        let vertex_alignment = self.backend_state.adapter_state.limits.min_vertex_input_binding_stride_alignment;
        let vertex_buffer_state = BufferState::new(
            &self.device_state,
            &vertices,
            vertex_alignment,
            65536,
            hal::buffer::Usage::VERTEX,
            &self.backend_state.adapter_state.memory_types,
        );

        let index_buffer_state = BufferState::new(
            &self.device_state,
            &indices,
            1,
            65536,
            hal::buffer::Usage::INDEX,
            &self.backend_state.adapter_state.memory_types,
        );

        self.vertex_buffer_state = Some(vertex_buffer_state);
        self.index_buffer_state = Some(index_buffer_state);
    }

    pub unsafe fn update_drawables(&mut self, mut drawables: Vec<Drawable>) {
        let ubos = drawables
            .iter()
            .map(|d| d.transform.to_ubo())
            .collect::<Vec<ObjectUniformBufferObject>>();
        println!("UBOs: {:?}", ubos);
        self.map_uniform_data(ubos);

        self.generate_image_states(
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

        println!("meshes by texture: {:?}", meshes_by_texture.iter().map(|(tex, meshes)| (tex.clone(), meshes.iter().map(|m| m.key.clone()).collect::<Vec<String>>())).collect::<Vec<(Option<Texture>, Vec<String>)>>());

        self.generate_cmd_buffers(meshes_by_texture);
        self.last_drawables = Some(drawables);
    }

    pub unsafe fn map_uniform_data(&mut self, ubos: Vec<ObjectUniformBufferObject>) {
        self.object_uniform
            .buffer
            .as_mut()
            .unwrap()
            .update_data(0, &ubos);
    }

    unsafe fn generate_image_states(&mut self, textures: Vec<&Texture>) {
        let new_textures: Vec<&Texture> = textures
            .into_iter()
            .filter(|t| !self.image_states.contains_key(&RenderKey::from(*t)))
            .unique()
            .collect();

        if new_textures.is_empty() {
            return;
        }

        for texture in new_textures.into_iter() {
            let image = Image::new(
                hal::buffer::Usage::TRANSFER_SRC,
                &texture.path,
                &hal::image::SamplerDesc::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp),
            );
            self.image_states.insert(RenderKey::from(texture), image_state);
        }

        self.device_state
            .read()
            .unwrap()
            .device
            .destroy_command_pool(staging_pool);
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
        let framebuffers = self.framebuffer_state
            .framebuffers
            .as_ref()
            .unwrap();

        let command_pools = self.framebuffer_state
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
            cmd_buffer.begin_primary(CommandBufferFlags::SIMULTANEOUS_USE);

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);

            cmd_buffer.bind_graphics_pipeline(&self.pipeline_state.pipeline.as_ref().unwrap());
            cmd_buffer.bind_vertex_buffers(0, Some((self.vertex_buffer_state.as_ref().unwrap().get_buffer(), 0)));
            cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                buffer: self.index_buffer_state.as_ref().unwrap().get_buffer(),
                offset: 0,
                index_type: hal::IndexType::U32
            });

            cmd_buffer.begin_render_pass(
                self.render_pass_state.render_pass.as_ref().unwrap(),
                &framebuffer,
                self.viewport.rect,
                &[
                    ClearValue { color: ClearColor { float32: [0.7, 0.2, 0.0, 1.0] } },
                    ClearValue { depth_stencil: ClearDepthStencil {depth: 1.0, stencil: 0} }
                ],
                SubpassContents::Inline
            );

            let mut current_mesh_index = 0;

            for (maybe_texture, meshes) in meshes_by_texture.iter() {
                let texture_key = RenderKey::from(maybe_texture);
                let texture_image_state = self.image_states.get(&texture_key).unwrap();

                cmd_buffer.bind_graphics_descriptor_sets(
                    &self.pipeline_state.pipeline_layout.as_ref().unwrap(),
                    2,
                    vec![ &texture_image_state.desc_set.descriptor_set ],
                    &[],
                );

                for (i, mesh) in meshes.iter().enumerate() {
                    if !mesh.rendered {
                        continue;
                    }

                    let dynamic_offset = i as u64 * self.object_uniform.buffer.as_ref().unwrap().padded_stride;

                    cmd_buffer.bind_graphics_descriptor_sets(
                        &self.pipeline_state.pipeline_layout.as_ref().unwrap(),
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

        self.framebuffer_state.command_buffers = Some(command_buffers);
    }
}

pub fn fps_view_matrix(eye: Vector3<f32>, pitch_rad: cgmath::Rad<f32>, yaw_rad: cgmath::Rad<f32>) -> Matrix4<f32> {
    use cgmath::Angle;

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

impl <B: hal::Backend> Drawer<B> for GfxDrawer<B> {
    fn draw(&mut self, image_index: u32) {
        let new_ubo = self.update_camera_uniform_buffer_object(dims, cmaer_transform);
        self.camera_uniform.buffer.as_mut().unwrap().update_data(0, &[new_ubo]);

        // TODO: THIS NEEDS TO BE REFACTORED
        let sem_index = self.framebuffers.next_acq_pre_pair_index();
        let (fid, sid) = self.framebuffers.get_frame_data(Some(frame_index), Some(sem_index));

        let (framebuffer_fence, command_buffer) = fid.unwrap();
        let (image_acquired_semaphore, image_present_semaphore) = sid.unwrap();

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

        let submission = Submission {
            command_buffers: std::iter::once(&*command_buffer),
            wait_semaphores: std::iter::once((&*image_acquired_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
            signal_semaphores: std::iter::once(&*image_present_semaphore),
        };

        self.device_state
            .write()
            .unwrap()
            .queue_group
            .queues[0]
            .submit(submission, Some(framebuffer_fence));
    }
}

struct RenderPass<B: hal::Backend> {
    render_pass: Option<B::RenderPass>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> RenderPass<B> {
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

impl<B: hal::Backend> Drop for RenderPass<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_render_pass(self.render_pass.take().unwrap());
        }
    }
}

struct Pipeline<B: hal::Backend> {
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> Pipeline<B> {
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
            let glsl = std::fs::read_to_string(data_path(vertex_shader)).unwrap();
            let mut spirv_file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex).unwrap();
            let spirv = hal::pso::read_spirv(&mut spirv_file).unwrap();
            device.create_shader_module(&spirv[..]).unwrap()
        };

        let fs_module = {
            let glsl = std::fs::read_to_string(data_path(fragment_shader)).unwrap();
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

impl<B: hal::Backend> Drop for Pipeline<B> {
    fn drop(&mut self) {
        let device = &self.device_state.read().unwrap().device;
        unsafe {
            device.destroy_graphics_pipeline(self.pipeline.take().unwrap());
            device.destroy_pipeline_layout(self.pipeline_layout.take().unwrap());
        }
    }
}

struct Framebuffers<B: hal::Backend> {
    framebuffers: Option<Vec<B::Framebuffer>>,
    framebuffer_fences: Option<Vec<B::Fence>>,
    command_pools: Option<Vec<B::CommandPool>>,
    command_buffers: Option<Vec<B::CommandBuffer>>,
    frame_images: Option<Vec<(B::Image, B::ImageView)>>,
    acquire_semaphores: Option<Vec<B::Semaphore>>,
    present_semaphores: Option<Vec<B::Semaphore>>,
    last_ref: usize,
    device_state: Arc<RwLock<DeviceState<B>>>,
    depth_image_stuff: Option<(B::Image, B::Memory, B::ImageView)>,
}

impl<B: hal::Backend> Framebuffers<B> {
    unsafe fn new(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        extent: Extent,
        images: Vec<B::Image>,
        image_format: hal::format::Format,
        render_pass_state: &RenderPass<B>,
        // TODO -> Get rid of / clean up
        depth_image_stuff: (B::Image, B::Memory, B::ImageView)
    ) -> Self
    {
        // TODO -> create depth image for each frame

        let (frame_images, framebuffers) = {
            let extent = hal::image::Extent {
                width: extent.width as _,
                height: extent.height as _,
                depth: 1,
            };

            let pairs = images
                .into_iter()
                .map(|image| {
                    let image_view = device_state
                        .read()
                        .unwrap()
                        .device
                        .create_image_view(
                            &image,
                            hal::image::ViewKind::D2,
                            image_format,
                            Swizzle::NO,
                            COLOR_RANGE.clone(),
                        )
                        .unwrap();

                    (image, image_view)
                })
                .collect::<Vec<_>>();

            let fbos = pairs
                .iter()
                .map(|&(_, ref image_view)| {
                    device_state
                        .read()
                        .unwrap()
                        .device
                        .create_framebuffer(
                            render_pass_state.render_pass.as_ref().unwrap(),
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
            fences.push(device_state.read().unwrap().device.create_fence(true).unwrap());
            command_pools.push(
                device_state
                    .read()
                    .unwrap()
                    .device
                    .create_command_pool(
                        device_state
                            .read()
                            .unwrap()
                            .queue_group
                            .family,
                        hal::pool::CommandPoolCreateFlags::empty(),
                    )
                    .expect("Can't create command pool"),
            );

            acquire_semaphores.push(device_state.read().unwrap().device.create_semaphore().unwrap());
            present_semaphores.push(device_state.read().unwrap().device.create_semaphore().unwrap());
        }

        Framebuffer {
            frame_images: Some(frame_images),
            framebuffers: Some(framebuffers),
            framebuffer_fences: Some(fences),
            command_pools: Some(command_pools),
            command_buffers: None,
            present_semaphores: Some(present_semaphores),
            acquire_semaphores: Some(acquire_semaphores),
            last_ref: 0,
            device_state: device_state.clone(),
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
        let device = &self.device_state.read().unwrap().device;

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
