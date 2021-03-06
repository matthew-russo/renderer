use std::sync::{Arc, RwLock};
use std::ops::Deref;
use std::path::Path;
use std::ffi::c_void;
use hal::window::{Extent2D, Surface};
use hal::device::Device;
use ash::vk;
use crate::renderer::core::RendererCore;
use crate::renderer::allocator::{Allocator, GfxAllocator};

pub const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };
const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;

type ImageIndex = u32;

pub(crate) trait Presenter<B: hal::Backend> : Send + Sync {
    fn images(&mut self) -> (Vec<B::Image>, hal::format::Format);
    fn semaphores(&mut self) -> (Option<&B::Semaphore>, Option<&B::Semaphore>);
    fn acquire_image(&mut self) -> Result<u32, String>;
    fn present(&mut self) -> Result<(), String>;
    fn viewport(&self) -> hal::pso::Viewport;
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

        let system_id = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        Self {
            instance,
            system_id
        }
    }

    pub fn create_vulkan_session(self, session_create_info: VulkanXrSessionCreateInfo) -> Result<VulkanXrSession, String> {
        unsafe {
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

            session
                .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
                .unwrap();

            Ok(VulkanXrSession {
                openxr: self,
                session,
                frame_waiter,
                frame_stream,
                swapchain: None,
                swapchain_format: hal::format::Format::Rgba8Srgb,
                swapchain_images: None,
                resolution: None,
                world_space: None,
            })
        }
    }
}

struct VulkanXrSession {
    openxr: OpenXr,
    session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    swapchain: Option<openxr::Swapchain<openxr::Vulkan>>,
    swapchain_format: hal::format::Format,
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

                // need to do this somewhere? VK_FORMAT_R8G8B8A8_SRGB
                gfx_backend_vulkan::native::Image {
                    raw: vk::Image::from_raw(*raw_image_ptr),
                    ty: vk::ImageType::TYPE_2D,
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

pub(crate) struct XrPresenter<B: hal::Backend, A: Allocator<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: Arc<RwLock<A>>,
    // TODO -> parameterize over graphics api
    vulkan_xr_session: VulkanXrSession,
    rendering_state: Option<openxr::FrameState>,
    acquired_image: Option<u32>,
    viewport: hal::pso::Viewport,
}

impl <B: hal::Backend> XrPresenter<B, GfxAllocator<B>> {
    fn session_create_info(core: &Arc<RwLock<RendererCore<B>>>) -> VulkanXrSessionCreateInfo {
        use ash::version::InstanceV1_0;
        let physical_device  = &core.read().unwrap().device.physical_device;
        let physical_device_any = physical_device as &dyn std::any::Any;
        let back_physical_device: &back::PhysicalDevice = physical_device_any.downcast_ref().unwrap();

        let gfx_device = &core.read().unwrap().device;
        let device = gfx_device.device.read().unwrap();
        let device_any = device.deref() as &dyn std::any::Any;
        let back_device: &back::Device = device_any.downcast_ref().unwrap();
        VulkanXrSessionCreateInfo {
            instance: back_physical_device.instance.0.handle(),
            physical_device: back_physical_device.handle,
            device: back_device.shared.raw.handle(),
            queue_family_index: core.read().unwrap().device.queue_family_id.unwrap().0 as u32,
            queue_index: 0,
        }
    }

    pub(crate) fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: &Arc<RwLock<GfxAllocator<B>>>) -> Self {
        let mut vulkan_xr_session = OpenXr::init()
            .create_vulkan_session(Self::session_create_info(core))
            .unwrap();
        vulkan_xr_session.create_swapchain();
        let viewport = Self::create_viewport(&vulkan_xr_session);

        Self {
            core: Arc::clone(core),
            allocator: Arc::clone(allocator),
            vulkan_xr_session,
            rendering_state: None,
            acquired_image: None,
            viewport,
        }
    }

    fn create_viewport(vulkan_xr_session: &VulkanXrSession) -> hal::pso::Viewport {
        hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: vulkan_xr_session.resolution.unwrap().width as _,
                h: vulkan_xr_session.resolution.unwrap().height as _,
            },
            depth: 0.0..1.0,
        }
    }
}

impl Presenter<gfx_backend_vulkan::Backend> for XrPresenter<gfx_backend_vulkan::Backend, GfxAllocator<gfx_backend_vulkan::Backend>> {
    fn images(&mut self) -> (Vec<gfx_backend_vulkan::native::Image>, hal::format::Format) {
        (
            self.vulkan_xr_session.swapchain_images.take().unwrap(),
            self.vulkan_xr_session.swapchain_format.clone()
        )
    }

    fn semaphores(&mut self) -> (Option<&gfx_backend_vulkan::native::Semaphore>, Option<&gfx_backend_vulkan::native::Semaphore>) {
        (None, None)
    }

    fn acquire_image(&mut self) -> Result<u32, String> {
        self.rendering_state = Some(self
            .vulkan_xr_session
            .frame_waiter
            .wait()
            .map_err(|e| e.to_string())?);

        let image = self
            .vulkan_xr_session
            .swapchain
            .as_mut()
            .unwrap()
            .acquire_image()
            .map_err(|e| e.to_string())?;

        self
            .vulkan_xr_session
            .swapchain
            .as_mut()
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
            .as_mut()
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
                                    .swapchain(&self.vulkan_xr_session.swapchain.as_ref().unwrap())
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
                                    .swapchain(self.vulkan_xr_session.swapchain.as_ref().unwrap())
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

    fn viewport(&self) -> hal::pso::Viewport {
        self.viewport.clone()
    }
}

pub(crate) struct MonitorPresenter<B: hal::Backend, A: Allocator<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: Arc<RwLock<A>>,
    swapchain: SxeSwapchain<B>,

    acquired_image: Option<ImageIndex>,
    viewport: hal::pso::Viewport,
}

impl <B: hal::Backend> MonitorPresenter<B, GfxAllocator<B>> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: &Arc<RwLock<GfxAllocator<B>>>) -> Self {
        let swapchain = SxeSwapchain::new(core);
        let viewport = Self::create_viewport(&swapchain);
        Self {
            core: Arc::clone(core),
            allocator: Arc::clone(allocator),
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

impl <B: hal::Backend> Presenter<B> for MonitorPresenter<B, GfxAllocator<B>> {
    fn images(&mut self) -> (Vec<B::Image>, hal::format::Format) {
        (
            self.swapchain.backbuffer.take().unwrap(),
            self.swapchain.format.clone()
        )
    }

    fn semaphores(&mut self) -> (Option<&B::Semaphore>, Option<&B::Semaphore>) {
        (
            Some(&self.swapchain.acquire_semaphores[self.swapchain.current_sem_index]),
            Some(&self.swapchain.present_semaphores[self.swapchain.current_sem_index]),
        )
    }

    fn acquire_image(&mut self) -> Result<u32, String> {
        use hal::window::Swapchain;

        if let Some(image_index) = self.acquired_image {
            return Err(format!("image {} already acquired without presenting", image_index));
        }

        let (image_index, _maybe_suboptimal) = self.swapchain.acquire_image()?;

        self.acquired_image = Some(image_index);

        Ok(image_index)
    }

    fn present(&mut self) -> Result<(), String> {
        let image_index = self
            .acquired_image
            .take()
            .ok_or(String::from("no image acquired to present to"))?;

        let queue = &mut self
            .core
            .write()
            .unwrap()
            .device
            .queue_group
            .queues[0];

        self.swapchain.present(queue, image_index)
    }

    fn viewport(&self) -> hal::pso::Viewport {
        self.viewport.clone()
    }
}

pub(crate) struct SxeSwapchain<B: hal::Backend> {
    pub core: Arc<RwLock<RendererCore<B>>>,
    pub swapchain: Option<B::Swapchain>,
    pub backbuffer: Option<Vec<B::Image>>,
    pub format: hal::format::Format,
    pub extent: hal::image::Extent,
    pub present_semaphores: Vec<B::Semaphore>,
    pub acquire_semaphores: Vec<B::Semaphore>,
    pub current_sem_index: usize,
}

impl<B: hal::Backend> SxeSwapchain<B> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        let caps = core
            .read()
            .unwrap()
            .backend
            .surface
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
            .capabilities(&core.read().unwrap().device.physical_device);

        let formats = core
            .read()
            .unwrap()
            .backend
            .surface
            .read()
            .unwrap()
            .as_ref()
            .unwrap()
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
            let surface_arc = Arc::clone(&core
                .write()
                .unwrap()
                .backend
                .surface);

            let mut writable_surface = surface_arc.write().unwrap();

            core
                .read()
                .unwrap()
                .device
                .device
                .read()
                .unwrap()
                .create_swapchain(writable_surface.as_mut().unwrap(), swap_config, None)
        }.expect("Can't create swapchain");

        // TODO -> this is duplicated in Drawer::new
        let iter_count = if backbuffer.len() != 0 {
            backbuffer.len()
        } else {
            1 // GL can have zero
        };

        let mut acquire_semaphores = vec![];
        let mut present_semaphores = vec![];

        for _ in 0..iter_count {
            acquire_semaphores.push(core.read().unwrap().device.device.read().unwrap().create_semaphore().unwrap());
            present_semaphores.push(core.read().unwrap().device.device.read().unwrap().create_semaphore().unwrap());
        }

        Self {
            core: Arc::clone(core),
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            format,
            extent,
            present_semaphores,
            acquire_semaphores,
            current_sem_index: 0,
        }
    }

    fn next_sem_index(&mut self) {
        self.current_sem_index += 1;

        if self.current_sem_index >= self.acquire_semaphores.len() {
            self.current_sem_index = 0
        }
    }

    pub fn acquire_image(&mut self) -> Result<(u32, Option<hal::window::Suboptimal>), String> {
        use hal::window::Swapchain;

        self.next_sem_index();
        let acquire_semaphore = &self.acquire_semaphores[self.current_sem_index];

        unsafe {
            self
                .swapchain
                .as_mut()
                .unwrap()
                .acquire_image(!0, Some(acquire_semaphore), None)
                .map_err(|e| e.to_string())
        }
    }

    pub fn present(&mut self, queue: &mut B::CommandQueue, image_index: u32) -> Result<(), String> {
        use hal::window::Swapchain;

        let present_semaphore = &self.present_semaphores[self.current_sem_index];

        unsafe {
            self
                .swapchain
                .as_ref()
                .unwrap()
                .present(
                    queue,
                    image_index,
                    Some(&*present_semaphore)
                )
                .map(|v| ())
                .map_err(|e| e.to_string())
        }
    }
}

impl <B: hal::Backend> Drop for SxeSwapchain<B> {
    fn drop(&mut self) {
        unsafe {
            let device_lock = &mut self.core.write().unwrap().device.device;
            let device = device_lock.write().unwrap();

            device.destroy_swapchain(self.swapchain.take().unwrap());

            for acquire_semaphore in self.acquire_semaphores.drain(..) {
                device.destroy_semaphore(acquire_semaphore);
            }

            for present_semaphore in self.present_semaphores.drain(..) {
                device.destroy_semaphore(present_semaphore);
            }
        }
    }
}
