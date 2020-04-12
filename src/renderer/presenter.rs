use std::sync::{Arc, RwLock};
use hal::window::{Extent2D};
use hal::pso::Viewport;
use hal::device::Device;
use hal::window::Surface;
use crate::renderer::core::RendererCore;
use crate::renderer::drawer::{Drawer, GfxDrawer};
use crate::renderer::allocator::Allocator;
use ash::vk;
use std::path::Path;
use std::ffi::c_void;

const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };
const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;

type ImageIndex = u32;

pub(crate) trait Presenter<B: hal::Backend> {
    fn acquire_image(&mut self, acquire_semaphore: B::Semaphore) -> Result<ImageIndex, String>;
    fn present(&mut self) -> Result<(), String>;
    fn viewport(&self) -> Viewport;
}


pub struct VulkanXrSessionCreateInfo {
    pub instance: vk::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: vk::Device,
    pub queue_family_index: u32,
    pub queue_index: u32,
}

struct OpenXr {
    instance: openxr::Instance,
    system_id: openxr::SystemId,
}

impl OpenXr {
    pub fn init() -> Self {
        let openxr_loader_path = Path::new("/usr/local/lib/libopenxr_loader.so");
        let entry = match openxr::Entry::load_from(openxr_loader_path) {
            Ok(e) => e,
            Err(e) => panic!("{:?}", e),
        };

        let extension_set = entry.enumerate_extensions().unwrap();

        let instance = entry.create_instance(
            &openxr::ApplicationInfo {
                application_name: "sxe test app",
                application_version: 0,
                engine_name: "sxe",
                engine_version: 0,
            },
            &extension_set,
        ).unwrap();

        let mut system_id = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        Self {
            instance,
            system_id
        }
    }

    pub unsafe fn create_vulkan_session(self, session_create_info: VulkanXrSessionCreateInfo) -> Result<VulkanXrSession, String> {
        use ash::vk::Handle;

        let create_info = openxr::vulkan::SessionCreateInfo {
            instance: session_create_info.instance.as_raw() as *const c_void,
            physical_device: session_create_info.physical_device.as_raw() as *const c_void,
            device: session_create_info.device.as_raw() as *const c_void,
            queue_family_index: session_create_info.queue_family_index,
            queue_index: session_create_info.queue_index,
        };

        let (session, frame_waiter, frame_stream) = self
            .instance
            .create_session(self.system_id, &create_info)
            .map_err(|e| e.to_string())?;

        Ok(VulkanXrSession {
            openxr: self,
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

struct VulkanXrSession {
    openxr: OpenXr,
    session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    swapchain: Option<openxr::Swapchain<openxr::Vulkan>>,
    swapchain_images: Option<Vec<gfx_backend_vulkan::native::Image>>,
    resolution: Option<openxr::Extent2Di>,
    world_space: Option<openxr::Space>,
}

impl VulkanXrSession {
    fn create_swapchain(&mut self) {
        let view_configuration_views = self.openxr.instance
            .enumerate_view_configuration_views(self.openxr.system_id, openxr::ViewConfigurationType::PRIMARY_STEREO)
            .unwrap();

        let resolution = openxr::Extent2Di {
            width: view_configuration_views[0].recommended_image_rect_width as i32,
            height: view_configuration_views[0].recommended_image_rect_height as i32,
        };

        let sample_count = view_configuration_views[0].recommended_swapchain_sample_count;

        let swapchain_create_info = openxr::SwapchainCreateInfo {
            create_flags: openxr::SwapchainCreateFlags::STATIC_IMAGE,
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


        let world_space = self.session
            .create_reference_space(openxr::ReferenceSpaceType::LOCAL, Self::default_pose())
            .unwrap();

        self.swapchain_images = Some(swapchain_images);
        self.swapchain = Some(swapchain);
        self.resolution = Some(resolution);
        self.world_space = Some(world_space);
    }

    fn draw(&mut self) {
        // if state.should_render {
        //     // draw scene
        //     renderer.draw(i);
        // }
    }

    fn default_pose() -> openxr::Posef {
        openxr::Posef {
            orientation: openxr::Quaternionf {
                x: 0.,
                y: 0.,
                z: 0.,
                w: 1.,
            },
            position: openxr::Vector3f {
                x: 0.,
                y: 0.,
                z: 0.,
            },
        }
    }
}

pub(crate) struct XrPresenter<B: hal::Backend, A: Allocator<B>, D: Drawer<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: A,
    drawer: D,
    // TODO -> parameterize over graphics api
    vulkan_xr_session: VulkanXrSession,
    rendering_state: Option<openxr::FrameState>,
    acquired_image: Option<u32>,
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> XrPresenter<B, A, D> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: A) -> Self {
        let vulkan_xr_session = OpenXr::init().create_vulkan_session();
        let drawer = GfxDrawer::new(core, allocator, viewport);

        Self {
            core: Arc::clone(core),
            allocator,
            drawer,
            vulkan_xr_session,
            rendering_state: None,
            acquired_image: None,
        }
    }
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> Presenter<B> for XrPresenter<B, A, D> {
    fn acquire_image(&mut self, acquire_semaphore: <B as Backend>::Semaphore) -> Result<u32, String> {
        self.rendering_state = Some(self
            .vulkan_xr_session
            .frame_waiter
            .wait()
            .map_err(|e| e.to_string())?);

        let image = self
            .vulkan_xr_session
            .swapchain
            .unwrap()
            .acquire_image()
            .map_err(|e| e.to_string())?;

        self
            .vulkan_xr_session
            .swapchain
            .unwrap()
            .wait_image(openxr::Duration::INFINITE)
            .map_err(|e| e.to_string())?;

        self
            .vulkan_xr_session
            .frame_stream
            .begin()
            .map_err(|e| e.to_string())?;

        self.acquired_image = Some(image);
        Ok(image)
    }

    fn present(&mut self) -> Result<(), String> {
        let (view_flags, views) = self.vulkan_xr_session.session
            .locate_views(
                openxr::ViewConfigurationType::PRIMARY_STEREO,
                self.rendering_state.unwrap().predicted_display_time,
                self.vulkan_xr_session.world_space.as_ref().unwrap(),
            )
            .map_err(|e| e.to_string())?;

        self.vulkan_xr_session
            .swapchain
            .unwrap()
            .release_image()
            .map_err(|e| e.to_string())?;

        self.vulkan_xr_session.frame_stream
            .end(
                self.rendering_state.unwrap().predicted_display_time,
                openxr::EnvironmentBlendMode::OPAQUE,
                &[&openxr::CompositionLayerProjection::new()
                    .space(self.vulkan_xr_session.world_space.as_ref().unwrap())
                    .views(&[
                        openxr::CompositionLayerProjectionView::new()
                            .pose(views[0].pose)
                            .fov(views[0].fov)
                            .sub_image(
                                openxr::SwapchainSubImage::new()
                                    .swapchain(&self.vulkan_xr_session.swapchain.unwrap())
                                    .image_array_index(0)
                                    .image_rect(openxr::Rect2Di {
                                        offset: openxr::Offset2Di { x: 0, y: 0 },
                                        extent: self.vulkan_xr_session.resolution.unwrap(),
                                    }),
                            ),
                        openxr::CompositionLayerProjectionView::new()
                            .pose(views[1].pose)
                            .fov(views[1].fov)
                            .sub_image(
                                openxr::SwapchainSubImage::new()
                                    .swapchain(&self.vulkan_xr_session.swapchain.unwrap())
                                    .image_array_index(1)
                                    .image_rect(openxr::Rect2Di {
                                        offset: openxr::Offset2Di { x: 0, y: 0 },
                                        extent: self.vulkan_xr_session.resolution.unwrap(),
                                    }),
                            )
                    ])]
            )
            .map_err(|e| e.to_string())
    }

    fn viewport(&self) -> Viewport {
        unimplemented!()
    }
}

pub(crate) struct MonitorPresenter<B: hal::Backend, A: Allocator<B>, D: Drawer<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: A,
    drawer: D,
    swapchain: SxeSwapchain<B>,
    acquired_image: Option<ImageIndex>,
    viewport: Viewport,
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> MonitorPresenter<B, A, D> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: A) -> Self {
        let swapchain = SxeSwapchain::new(core);
        let viewport = Self::create_viewport(&swapchain);
        let drawer: D = GfxDrawer::new(core, allocator, viewport);
        Self {
            core: Arc::clone(core),
            allocator,
            drawer,
            swapchain,
            acquired_image: None,
            viewport,
        }
    }

    fn create_viewport(swapchain_state: &SxeSwapchain<B>) -> hal::pso::Viewport {
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
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> Presenter<B> for MonitorPresenter<B, A, D> {
    fn acquire_image(&mut self, acquire_semaphore: B::Semaphore) -> Result<u32, String> {
        use hal::window::Swapchain;

        if let Some(image_index) = self.acquired_image {
            return Err(format!("image {} already acquired without presenting", image_index));
        }

        let (image_index, maybe_suboptimal) = unsafe {
            self
                .swapchain
                .swapchain
                .unwrap()
                .acquire_image(!0, Some(&acquire_semaphore), None)
                .map_err(|e| e.to_string())?
        };

        self.acquired_image = Some(image_index);

        Ok(image_index)
    }

    fn present(&mut self, image_present_semaphore: &B::Semaphore) -> Result<(), String> {
        use hal::window::Swapchain;

        unsafe {
            let image_index = self
                .acquired_image
                .take()
                .ok_or(String::from("no image acquired to present to"))?;

            let mut queue = self
                .core
                .read()
                .unwrap()
                .device
                .queue_group
                .queues[0];

            self.swapchain
                .swapchain
                .unwrap()
                .present(
                    &mut queue,
                    image_index,
                    Some(&*image_present_semaphore)
                )
                .map(|v| ())
                .map_err(|e| e.to_string())
        }
    }

    fn viewport(&self) -> Viewport {
        self.viewport.clone()
    }
}

pub(crate) struct SxeSwapchain<B: hal::Backend> {
    pub core: Arc<RwLock<RendererCore<B>>>,
    pub swapchain: Option<B::Swapchain>,
    pub backbuffer: Option<Vec<B::Image>>,
    pub format: hal::format::Format,
    pub extent: hal::image::Extent,
}

impl<B: hal::Backend> SxeSwapchain<B> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        let caps = core
            .read()
            .unwrap()
            .backend
            .surface
            .capabilities(&core.read().unwrap().device.physical_device);

        let formats = core
            .read()
            .unwrap()
            .backend
            .surface
            .supported_formats(&core.read().unwrap().device.physical_device);

        let format = formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == hal::format::ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = hal::window::SwapchainConfig::from_caps(&caps, format, DIMS);

        let extent = swap_config.extent.to_extent();

        let (swapchain, backbuffer) = unsafe {
            core
                .write()
                .unwrap()
                .device
                .device
                .create_swapchain(&mut core.read().unwrap().backend.surface, swap_config, None)
        }.expect("Can't create swapchain");

        Self {
            core: Arc::clone(core),
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            format,
            extent,
        }
    }
}

impl<B: hal::Backend> Drop for SxeSwapchain<B> {
    fn drop(&mut self) {
        unsafe {
            self.core
                .read()
                .unwrap()
                .device
                .device
                .destroy_swapchain(self.swapchain.take().unwrap());
        }
    }
}
