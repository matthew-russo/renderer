use std::sync::{Arc, RwLock};
use std::collections::{HashMap};

use hal::adapter::{MemoryType};
use hal::device::{Device};
use hal::format::{Format, Swizzle, Aspects};
use hal::pso::{ShaderStageFlags};
use hal::window::{Extent2D};

use cgmath::{
    Vector3,
    Matrix4,
    SquareMatrix,
};

use crate::primitives::uniform_buffer_object::{
    CameraUniformBufferObject,
    ObjectUniformBufferObject,
};
use crate::components::transform::Transform;
use crate::renderer::render_key::RenderKey;
use crate::xr::xr::VulkanXrSessionCreateInfo;

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
        let base_image_state = ImageState::new(
            image_desc_set,
            &device_state,
            &backend_state.adapter_state,
            hal::buffer::Usage::TRANSFER_SRC,
            &mut staging_pool,
            "textures/chalet.jpg",
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
            backend_state,
            device_state,

            recreate_swapchain: false,
            resize_dims,
        }
    }

    pub unsafe fn draw_frame(&mut self, camera_transform: &Transform) {
        // if self.recreate_swapchain {
        //     self.recreate_swapchain();
        //     self.recreate_swapchain = false;
        // }
        // let dims = [DIMS.width as f32, DIMS.height as f32];

        // ...
        // ...
        // ...

        // present frame
        // if let Err(e) = self
        //     .swapchain_state
        //     .swapchain
        //     .as_mut()
        //     .unwrap()
        //     .present(
        //         &mut self.device_state.write().unwrap().queue_group.queues[0],
        //         frame,
        //         Some(&*image_present_semaphore),
        //     )
        // {
        //     println!("error on presenting: {:?}", e);
        //     self.recreate_swapchain = true;
        // }
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
    pub fn vulkan_session_create_info(&self) -> Result<VulkanXrSessionCreateInfo, String> {
        Err(String::from("trying to create VulkanXrSessionCreateInfo while not using vulkan"))
    }

    #[cfg(feature="vulkan")]
    pub unsafe fn vulkan_session_create_info(&self) -> Result<VulkanXrSessionCreateInfo, String> {
        use ash::version::InstanceV1_0;
        let physical_device  = &self.device_state.read().unwrap().physical_device;
        let physical_device_any = physical_device as &dyn std::any::Any;
        let back_physical_device: &back::PhysicalDevice = physical_device_any.downcast_ref().unwrap();

        let device = &self.device_state.read().unwrap().device;
        let device_any = device as &dyn std::any::Any;
        let back_device: &back::Device = device_any.downcast_ref().unwrap();
        Ok(VulkanXrSessionCreateInfo {
            instance: back_physical_device.instance.0.handle(),
            physical_device: back_physical_device.handle,
            device: back_device.raw.0.handle(),
            queue_family_index: self.device_state.read().unwrap().queue_family_index.unwrap(),
            queue_index: 0,
        })
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
