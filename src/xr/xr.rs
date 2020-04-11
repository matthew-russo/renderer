use std::path::Path;
use ash::vk;
use openxr::{ApplicationInfo, Entry, FormFactor, FrameWaiter, FrameStream, Instance, SystemId, Session, Vulkan, Result as OpenXrResult, vulkan::SessionCreateInfo, Swapchain, SwapchainCreateInfo, SwapchainCreateFlags, ViewConfigurationType, Space, Posef, Quaternionf, Vector3f, ReferenceSpaceType, Extent2Di};
use std::ffi::c_void;

const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;

pub(crate) struct Xr {
    instance: Instance,
    system_id: SystemId,
}

impl Xr {
    pub fn init() -> Self {
        let openxr_loader_path = Path::new("/usr/local/lib/libopenxr_loader.so");
        let entry = match Entry::load_from(openxr_loader_path) {
            Ok(e) => e,
            Err(e) => panic!("{:?}", e),
        };

        let extension_set = entry.enumerate_extensions().unwrap();

        let instance = entry.create_instance(
            &ApplicationInfo {
                application_name: "sxe test app",
                application_version: 0,
                engine_name: "sxe",
                engine_version: 0,
            },
            &extension_set,
        ).unwrap();

        let mut system_id = instance.system(FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        Self {
            instance,
            system_id
        }
    }

    pub unsafe fn create_vulkan_session(self, session_create_info: VulkanXrSessionCreateInfo) -> OpenXrResult<VulkanXrSession> {
        use ash::vk::Handle;

        let create_info = SessionCreateInfo {
            instance: session_create_info.instance.as_raw() as *const c_void,
            physical_device: session_create_info.physical_device.as_raw() as *const c_void,
            device: session_create_info.device.as_raw() as *const c_void,
            queue_family_index: session_create_info.queue_family_index,
            queue_index: session_create_info.queue_index,
        };

        let (session, frame_waiter, frame_stream) = self.instance.create_session(self.system_id, &create_info)?;

        Ok(VulkanXrSession {
            xr: self,
            session,
            frame_waiter,
            frame_stream,
            swapchain: None,
            swapchain_images: None,
            resolution: None,
            world_space: None,
        })
    }
}

pub struct VulkanXrSessionCreateInfo {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue_family_index: u32,
    pub queue_index: u32,
}

pub struct VulkanXrSession {
    xr: Xr,
    session: Session<Vulkan>,
    frame_waiter: FrameWaiter,
    frame_stream: FrameStream<Vulkan>,
    swapchain: Option<Swapchain<Vulkan>>,
    swapchain_images: Option<Vec<gfx_backend_vulkan::native::Image>>,
    resolution: Option<Extent2Di>,
    world_space: Option<Space>,
}

impl VulkanXrSession {
    fn create_swapchain(&mut self) {
        let view_configuration_views = self.xr.instance
            .enumerate_view_configuration_views(self.xr.system_id, ViewConfigurationType::PRIMARY_STEREO)
            .unwrap();

        let resolution = Extent2Di {
            width: view_configuration_views[0].recommended_image_rect_width as i32,
            height: view_configuration_views[0].recommended_image_rect_height as i32,
        };

        let sample_count = view_configuration_views[0].recommended_swapchain_sample_count;

        let swapchain_create_info = SwapchainCreateInfo {
            create_flags: SwapchainCreateFlags::STATIC_IMAGE,
            usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | openxr::SwapchainUsageFlags::SAMPLED,
            format: VK_FORMAT_R8G8B8A8_SRGB,
            sample_count,
            width: resolution.width as u32,
            height: resolution.height as u32,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        };

        let swapchain = self.session.create_swapchain(&swapchain_create_info).unwrap();
        let swapchain_images = swapchain
            .enumerate_images()
            .unwrap()
            .iter()
            .map(|raw_image_ptr| {
                use ash::vk::Handle;

                gfx_backend_vulkan::native::Image {
                    raw: vk::Image::from_raw(*raw_image_ptr),
                    ty: vk::ImageType::from_raw(VK_FORMAT_R8G8B8A8_SRGB as i32),
                    flags: vk::ImageCreateFlags::empty(),
                    extent: vk::Extent3D {
                        width: resolution.width as u32,
                        height: resolution.height as u32,
                        depth: 1,
                    },
                }
            })
            .collect();

        let pose = Posef {
            orientation: Quaternionf {
                x: 0.,
                y: 0.,
                z: 0.,
                w: 1.,
            },
            position: Vector3f {
                x: 0.,
                y: 0.,
                z: 0.,
            },
        };
        let world_space = self.session
            .create_reference_space(ReferenceSpaceType::LOCAL, pose)
            .unwrap();

        self.swapchain_images = Some(swapchain_images);
        self.swapchain = Some(swapchain);
        self.resolution = Some(resolution);
        self.world_space = Some(world_space);
    }

    fn draw(&mut self) {
        let state = self.frame_waiter.wait().unwrap();
        let image = self.swapchain.unwrap().acquire_image().unwrap();
        self.swapchain.unwrap().wait_image(openxr::Duration::INFINITE).unwrap();

        self.frame_stream.begin().unwrap();

        if state.should_render {
            // draw scene
            renderer.draw(i);
        }

        let (view_flags, views) = self.session
            .locate_views(
                openxr::ViewConfigurationType::PRIMARY_STEREO,
                state.predicted_display_time,
                self.world_space.as_ref().unwrap(),
            )
            .unwrap();

        self.swapchain.unwrap().release_image().unwrap();
        self.frame_stream
            .end(
                state.predicted_display_time,
                openxr::EnvironmentBlendMode::OPAQUE,
                &[&openxr::CompositionLayerProjection::new()
                    .space(self.world_space.as_ref().unwrap())
                    .views(&[
                        openxr::CompositionLayerProjectionView::new()
                            .pose(views[0].pose)
                            .fov(views[0].fov)
                            .sub_image(
                                openxr::SwapchainSubImage::new()
                                    .swapchain(&self.swapchain.unwrap())
                                    .image_array_index(0)
                                    .image_rect(openxr::Rect2Di {
                                        offset: openxr::Offset2Di { x: 0, y: 0 },
                                        extent: self.resolution.unwrap(),
                                    }),
                            ),
                        openxr::CompositionLayerProjectionView::new()
                            .pose(views[1].pose)
                            .fov(views[1].fov)
                            .sub_image(
                                openxr::SwapchainSubImage::new()
                                    .swapchain(&self.swapchain.unwrap())
                                    .image_array_index(1)
                                    .image_rect(openxr::Rect2Di {
                                        offset: openxr::Offset2Di { x: 0, y: 0 },
                                        extent: self.resolution.unwrap(),
                                    }),
                            )
                    ])]
            )
            .unwrap();
    }
}
