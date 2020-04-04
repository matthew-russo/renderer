use std::fs;
use std::fs::File;
use std::sync::{Arc, RwLock};
use std::io::{BufReader};
use std::collections::{HashMap, BTreeMap};
use std::path::{Path, PathBuf};

use itertools::{Itertools};

use glsl_to_spirv;

use hal::{Limits, Instance};
use hal::adapter::{Adapter, MemoryType, PhysicalDevice};
use hal::command::{CommandBuffer, CommandBufferFlags, ClearColor, ClearDepthStencil, ClearValue, SubpassContents};
use hal::device::{Device};
use hal::format::{Format, AsFormat, ChannelType, Rgba8Srgb, Swizzle, Aspects};
use hal::pass::Subpass;
use hal::pso::{DescriptorPool, PipelineStage, ShaderStageFlags, VertexInputRate, Viewport};
use hal::pool::{CommandPool};
use hal::queue::{CommandQueue, Submission, QueueFamily};
use hal::queue::family::QueueGroup;
use hal::window::{Extent2D, Surface, Swapchain, SwapchainConfig};

use image::load as load_image;

use cgmath::{
    Deg,
    Vector3,
    Matrix4,
    SquareMatrix,
    perspective,
};

use crate::primitives::vertex::Vertex;

use crate::components::mesh::Mesh;
use crate::primitives::uniform_buffer_object::{
    CameraUniformBufferObject,
    ObjectUniformBufferObject,
};
use crate::primitives::drawable::Drawable;
use crate::components::transform::Transform;
use crate::components::texture::Texture;
use crate::renderer::render_key::RenderKey;
use crate::xr::xr::VulkanXrSessionCreateInfo;

pub(crate) const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

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

fn data_path(specific_file: &str) -> PathBuf {
    let root_data = Path::new("src/data");
    let specific_file_path = Path::new(specific_file);
    return root_data.join(specific_file_path);
}

struct ImageData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: hal::format::Format,
}

pub struct Renderer<B: hal::Backend> {
    backend_state: BackendState<B>,
    device_state: Arc<RwLock<DeviceState<B>>>,

    recreate_swapchain: bool,
    resize_dims: Extent2D,
}

impl<B: hal::Backend> Renderer<B> {
   pub unsafe fn new(mut backend_state: BackendState<B>) -> Self {
        let device_state = Arc::new(
            RwLock::new(
                DeviceState::new(
                    backend_state.adapter_state.adapter.take().unwrap(),
                    &backend_state.surface
                )
            )
        );

        let camera_uniform_desc_set_layout = Arc::new(RwLock::new(DescSetLayout::new(
            &device_state,
            vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Buffer {
                    ty: hal::pso::BufferDescriptorType::Uniform,
                    format: hal::pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    }
                },
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }]
        )));

        let camera_uniform_desc_set = Self::create_set(&camera_uniform_desc_set_layout, &mut uniform_desc_pool);

        let object_uniform_desc_set_layout = Arc::new(RwLock::new(DescSetLayout::new(
            &device_state,
            vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Buffer {
                    ty: hal::pso::BufferDescriptorType::Uniform,
                    format: hal::pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: true,
                    }
                },
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
                immutable_samplers: false,
            }]
        )));

        let object_uniform_desc_set = Self::create_set(&object_uniform_desc_set_layout, &mut uniform_desc_pool);

        let image_desc_set_layout = Arc::new(RwLock::new(DescSetLayout::new(
            &device_state,
            vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Image {
                    ty: hal::pso::ImageDescriptorType::Sampled {
                        with_sampler: true,
                    }
                },
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            }]
        )));

        let image_desc_set = Self::create_set(&image_desc_set_layout, &mut image_desc_pool);
        let mut staging_pool = device_state
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
            .expect("Can't create staging command pool");

        let row_alignment_mask = backend_state.adapter_state.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
        let image_data = load_image_data("textures/chalet.jpg", row_alignment_mask);
        let base_image_state = ImageState::new(
            image_desc_set,
            &device_state,
            &backend_state.adapter_state,
            hal::buffer::Usage::TRANSFER_SRC,
            &mut staging_pool,
            &image_data,
            &hal::image::SamplerDesc::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp),
        );

        device_state
            .read()
            .unwrap()
            .device
            .destroy_command_pool(staging_pool);

        let mut image_states = HashMap::new();
        image_states.insert(RenderKey::from(&None), base_image_state);

        // TODO -> merge the camera transform and ubo initialization

        let font_tex_desc_set_layout = Arc::new(RwLock::new(DescSetLayout::new(
            &device_state,
            vec![hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            }]
        )));

        let my_temp_view = Matrix4::look_at(
            cgmath::Point3::new(5.0, 5.0, 5.0),
            cgmath::Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0)
        );
        let mut my_temp_proj = perspective(
            Deg(45.0),
            DIMS.width as f32 / DIMS.height as f32,
            0.1,
            1000.0
        );
        my_temp_proj.y.y *= -1.0;

        let camera_uniform_buffer_object = CameraUniformBufferObject::new(
            my_temp_view,
            my_temp_proj
        );
        let camera_uniform = Uniform::new(
            &backend_state.adapter_state,
            &device_state,
            &[camera_uniform_buffer_object],
            camera_uniform_desc_set,
            0
        );

        let object_uniform_buffer_object = ObjectUniformBufferObject::new(
            Matrix4::identity(),
        );
        let object_uniform = Uniform::new(
            &backend_state.adapter_state,
            &device_state,
            &[object_uniform_buffer_object, object_uniform_buffer_object],
            object_uniform_desc_set,
            0
        );

        let depth_image_stuff = create_image_stuff::<B>(
            &device_state.read().unwrap().device,
            &backend_state.adapter_state.memory_types,
            swapchain_state.extent.width,
            swapchain_state.extent.height,
            hal::format::Format::D32SfloatS8Uint,
            hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            Aspects::DEPTH | Aspects::STENCIL
        );

        let resize_dims = Extent2D {
            width: DIMS.width,
            height: DIMS.height,
        };
        
        Self {
            image_desc_pool: Some(image_desc_pool),
            uniform_desc_pool: Some(uniform_desc_pool),
            font_tex_desc_pool: Some(font_tex_desc_pool),
            viewport,

            backend_state,
            device_state,
            swapchain_state,
            render_pass_state,
            pipeline_state,
            framebuffer_state,

            image_desc_set_layout: Some(image_desc_set_layout),
            image_states,

            vertex_buffer_state: None,
            index_buffer_state: None,
            camera_uniform,
            object_uniform,

            recreate_swapchain: false,
            resize_dims,
            last_drawables: None,
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        self.backend_state.window()
    }

    // Pitch must be in the range of [-90 ... 90] degrees and 
    // yaw must be in the range of [0 ... 360] degrees.
    // Pitch and yaw variables must be expressed in radians.
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

    pub fn update_camera_uniform_buffer_object(&self, dimensions: [f32;2], camera_transform: &Transform) -> CameraUniformBufferObject {
        let position = camera_transform.position;
        let rotation = cgmath::Euler::from(camera_transform.rotation);

        let view = Self::fps_view_matrix(position, rotation.y, rotation.x);

        // let view = Matrix4::look_at(
        //     Point3::new(position.x, position.y, position.z),
        //     Point3::new(
        //         Deg::from(rotation.x).0,
        //         Deg::from(rotation.y).0,
        //         Deg::from(rotation.z).0
        //     ),
        //     Vector3::new(0.0, 1.0, 0.0)
        // );

        let mut proj = perspective(
            Deg(45.0),
            dimensions[0] / dimensions[1],
            0.1,
            1000.0
        );

        proj.y.y *= -1.0;

        CameraUniformBufferObject::new(view, proj)
    }

    // // TODO -> change this to just map_uniform_data and pass in the uniform we're targeting
    // pub unsafe fn map_object_uniform_data(&mut self, uniform_data: Vec<ObjectUniformBufferObject>) {
    //     let device_writable = &mut self.device_state.write().unwrap().device;

    //     // TODO -> Pass in the uniform that we need
    //     let uniform_buffer = self
    //         .object_uniform
    //         .buffer
    //         .as_mut()
    //         .unwrap();

    //     let uniform_memory = uniform_buffer
    //         .buffer_memory
    //         .as_ref()
    //         .unwrap();

    //     if uniform_buffer.memory_is_mapped {
    //         device_writable.unmap_memory(uniform_memory);
    //         uniform_buffer.memory_is_mapped = false;
    //     }

    //     match device_writable.map_memory(uniform_memory, 0..uniform_buffer.size) {
    //         Ok(mem_ptr) => {
    //             // if !coherent {
    //             //     device.invalidate_mapped_memory_ranges(
    //             //         Some((
    //             //            buffer.memory(),
    //             //            range.clone()
    //             //         ))
    //             //     );
    //             // }

    //             let data_as_bytes = uniform_data
    //                 .iter()
    //                 .flat_map(|ubo| any_as_u8_slice(ubo, padded_stride))
    //                 .collect::<Vec<u8>>();

    //             let slice = std::slice::from_raw_parts_mut(mem_ptr, data_as_bytes.len());
    //             slice.copy_from_slice(&data_as_bytes[..]);
    //
    //             // if !coherent {
    //             //     device.flush_mapped_memory_ranges(
    //             //         Some((
    //             //             buffer.memory(),
    //             //             range
    //             //         ))
    //             //     );
    //             // }
    //         },
    //         Err(e) => panic!("error mapping memory: {:?}", e),
    //     }

    //     uniform_buffer.memory_is_mapped = true;
    // }

    fn create_set(desc_set_layout: &Arc<RwLock<DescSetLayout<B>>>, descriptor_pool: &mut B::DescriptorPool) -> DescSet<B> {
        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(desc_set_layout.read().unwrap().layout.as_ref().unwrap())
        }.unwrap();

        DescSet {
            descriptor_set,
            desc_set_layout: desc_set_layout.clone()
        }
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

        for texture in new_textures.into_iter() {
            let image_desc_set = Self::create_set(self.image_desc_set_layout.as_ref().unwrap(), self.image_desc_pool.as_mut().unwrap());
            let row_alignment_mask = self.backend_state.adapter_state.limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
            let image_data = load_image_data(&texture.path, row_alignment_mask);
            let image_state = ImageState::new(
                image_desc_set,
                &self.device_state,
                &self.backend_state.adapter_state,
                hal::buffer::Usage::TRANSFER_SRC,
                &mut staging_pool,
                &image_data,
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

    pub unsafe fn map_uniform_data(&mut self, ubos: Vec<ObjectUniformBufferObject>) {
        self.object_uniform
            .buffer
            .as_mut()
            .unwrap()
            .update_data(0, &ubos);
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

    pub unsafe fn draw_frame(&mut self, camera_transform: &Transform) {
        if self.recreate_swapchain {
            self.recreate_swapchain();
            self.recreate_swapchain = false; 
        }

        let dims = [DIMS.width as f32, DIMS.height as f32];

        let new_ubo = self.update_camera_uniform_buffer_object(dims, camera_transform);
        self.camera_uniform.buffer.as_mut().unwrap().update_data(0, &[new_ubo]);

        let sem_index = self.framebuffer_state.next_acq_pre_pair_index();

        let frame: hal::window::SwapImageIndex = {
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
					error!("we gots an error on AQUIREIMAGE: {:?}", e);
                    self.recreate_swapchain = true;
                    return;
                }
            }
        };

        let (fid, sid) = self.framebuffer_state
            .get_frame_data(Some(frame as usize), Some(sem_index));

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
                Some(&*image_present_semaphore),
            )
        {
            println!("error on presenting: {:?}", e);
            self.recreate_swapchain = true;
        }
    }

    unsafe fn recreate_swapchain(&mut self) {
        println!("\n\nrecreating swapchain\n\n");

        self.device_state.read().unwrap().device.wait_idle().unwrap();

        self.swapchain_state = SwapchainState::new(&mut self.backend_state, &self.device_state);

        self.render_pass_state = RenderPassState::new(&self.device_state, &self.swapchain_state);

        let depth_image_stuff = create_image_stuff::<B>(
            &self.device_state.read().unwrap().device,
            &self.backend_state.adapter_state.memory_types,
            self.swapchain_state.extent.width,
            self.swapchain_state.extent.height,
            hal::format::Format::D32SfloatS8Uint,
            hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            hal::format::Aspects::DEPTH | Aspects::STENCIL
        );

        self.framebuffer_state = FramebufferState::new(
            &self.device_state,
            &mut self.swapchain_state,
            &self.render_pass_state,
            depth_image_stuff
        );

        self.pipeline_state = PipelineState::new(
            &self.device_state,
            self.render_pass_state.render_pass.as_ref().unwrap(),
            vec![
                self.camera_uniform.desc.as_ref().unwrap().desc_set_layout.read().unwrap().layout.as_ref().unwrap(),
                self.object_uniform.desc.as_ref().unwrap().desc_set_layout.read().unwrap().layout.as_ref().unwrap(),
                self.image_desc_set_layout.as_ref().unwrap().read().unwrap().layout.as_ref().unwrap()
            ],
            "shaders/standard.vert",
            "shaders/standard.frag",
        );

        self.viewport = Self::create_viewport(&self.swapchain_state);
        let drawables = self.last_drawables.take().unwrap();
        self.update_drawables(drawables);
    }

    #[cfg(not(feature="vulkan"))]
    pub fn vulkan_session_create_info(&self) -> VulkanXrSessionCreateInfo {
        panic!("trying to create VulkanXrSessionCreateInfo while not using vulkan");
    }

    #[cfg(feature="vulkan")]
    pub unsafe fn vulkan_session_create_info(&self) -> VulkanXrSessionCreateInfo {
        use ash::version::InstanceV1_0;
        let physical_device  = &self.device_state.read().unwrap().physical_device;
        let physical_device_any = physical_device as &dyn std::any::Any;
        let back_physical_device: &back::PhysicalDevice = physical_device_any.downcast_ref().unwrap();

        let device = &self.device_state.read().unwrap().device;
        let device_any = device as &dyn std::any::Any;
        let back_device: &back::Device = device_any.downcast_ref().unwrap();
        VulkanXrSessionCreateInfo {
            instance: back_physical_device.instance.0.handle(),
            physical_device: back_physical_device.handle,
            device: back_device.raw.0.handle(),
            queue_family_index: self.device_state.read().unwrap().queue_family_index.unwrap(),
            queue_index: 0,
        }
    }
}

// TODO -> get rid of this
/// Create an image, image memory, and image view with the given properties.
pub unsafe fn create_image_stuff<B: hal::Backend>(
    device: &B::Device,
    memory_types: &[MemoryType],
    width: u32,
    height: u32,
    format: Format,
    usage: hal::image::Usage,
    aspects: Aspects,
) -> (B::Image, B::Memory, B::ImageView) {
    let kind = hal::image::Kind::D2(width, height, 1, 1);

    let mut image = device
        .create_image(
            kind,
            1,
            format,
            hal::image::Tiling::Optimal,
            usage,
            hal::image::ViewCapabilities::empty(),
        )
        .expect("Failed to create unbound image");

    let image_req = device.get_image_requirements(&image);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            image_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();

    let image_memory = device
        .allocate_memory(device_type, image_req.size)
        .expect("Failed to allocate image");

    device
        .bind_image_memory(&image_memory, 0, &mut image)
        .expect("Failed to bind image");

    let image_view = device
        .create_image_view(
            &image,
            hal::image::ViewKind::D2,
            format,
            Swizzle::NO,
            hal::image::SubresourceRange {
                aspects,
                levels: 0..1,
                layers: 0..1,
            },
        )
        .expect("Failed to create image view");

    (image, image_memory, image_view)
}

impl<B: hal::Backend> Drop for Renderer<B> {
    fn drop(&mut self) {
        self.device_state
            .read()
            .unwrap()
            .device
            .wait_idle()
            .unwrap();

        unsafe {
            self.device_state
                .read()
                .unwrap()
                .device
                .destroy_descriptor_pool(self.image_desc_pool.take().unwrap());

            self.device_state
                .read()
                .unwrap()
                .device
                .destroy_descriptor_pool(self.uniform_desc_pool.take().unwrap());
        }
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T, pad_to_size: usize) -> Vec<u8> {
    let mut vec = std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        std::mem::size_of::<T>())
        .to_vec();

    vec.resize(pad_to_size, 0);

    vec
}
