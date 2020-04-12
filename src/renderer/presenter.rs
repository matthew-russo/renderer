use std::sync::{Arc, RwLock};
use hal::window::{Extent2D};
use hal::pso::Viewport;
use hal::device::Device;
use hal::window::Surface;
use crate::renderer::core::RendererCore;
use crate::renderer::drawer::{Drawer, GfxDrawer};
use crate::renderer::allocator::Allocator;

const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };

type ImageIndex = u32;

pub(crate) trait Presenter<B: hal::Backend> {
    fn acquire_image(&mut self, acquire_semaphore: B::Semaphore) -> Result<ImageIndex, String>;
    fn present(&mut self) -> Result<(), String>;
    fn viewport(&self) -> Viewport;
}

pub(crate) struct XrPresenter<B: hal::Backend, A: Allocator<B>, D: Drawer<B>> {
    core: RendererCore<B>,
    allocator: A,
    drawer: D,
    session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    // TODO -> parameterize Graphics API
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    swapchain: Option<openxr::Swapchain<openxr::Vulkan>>,
    swapchain_images: Option<Vec<gfx_backend_vulkan::native::Image>>,

    rendering_state: Option<openxr::FrameState>,
    acquired_image: Option<u32>,
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> XrPresenter<B, A, D> {
    fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: A) -> Self {

    }
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> Presenter<B> for XrPresenter<B, A, D> {
    fn acquire_image(&mut self, acquire_semaphore: <B as Backend>::Semaphore) -> Result<u32, String> {
        self.rendering_state = Some(self.frame_waiter.wait()?);
        let image = self.swapchain.unwrap().acquire_image()?;
        self.swapchain.unwrap().wait_image(openxr::Duration::INFINITE)?;
        self.frame_stream.begin()?;
        self.acquired_image = Some(image);
        Ok(image)
    }

    fn present(&mut self) -> Result<(), String> {
        let (view_flags, views) = self.session
            .locate_views(
                openxr::ViewConfigurationType::PRIMARY_STEREO,
                self.rendering_state.unwrap().predicted_display_time,
                self.world_space.as_ref().unwrap(),
            )
            .unwrap();

        self.swapchain.unwrap().release_image().unwrap();
        self.frame_stream
            .end(
                self.rendering_state.unwrap().predicted_display_time,
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

    fn viewport(&self) -> Viewport {
        unimplemented!()
    }
}

pub(crate) struct MonitorPresenter<B: hal::Backend, A: Allocator<B>, D: Drawer<B>> {
    core: Arc<RwLock<RendererCore<B>>>,
    allocator: A,
    drawer: D,
    swapchain: Swapchain<B>,
    acquired_image: Option<ImageIndex>,
    viewport: Viewport,
}

impl <B: hal::Backend, A: Allocator<B>, D: Drawer<B>> MonitorPresenter<B, A, D> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>, allocator: A) -> Self {
        let drawer: D = GfxDrawer::new(core);
        let swapchain = Swapchain::new(core);
        let viewport = Self::create_viewport(&swapchain);
        Self {
            core: Arc::clone(core),
            allocator,
            drawer,
            swapchain,
            acquired_image: None,
            viewport,
        }
    }

    fn create_viewport(swapchain_state: &Swapchain<B>) -> hal::pso::Viewport {
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
        if let Some(image_index) = self.acquired_image {
            return Err(format!("image {} already acquired without presenting", image_index));
        }

        let image_index = self
            .swapchain
            .swapchain
            .unwrap()
            .acquire_image(!0, Some(acquire_semaphore), None)?;

        self.acquired_image = Some(image_index);

        Ok(image_index)
    }

    fn present(&mut self, image_present_semaphore: &B::Semaphore) -> Result<(), String> {
        unsafe {
            let image_index = self
                .acquired_image
                .take()
                .ok_or(String::from("no image acquired to present to"))?;

            let queue = self
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
                    queue,
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

pub(crate) struct Swapchain<B: hal::Backend> {
    pub core: Arc<RwLock<RendererCore<B>>>,
    pub swapchain: Option<B::Swapchain>,
    pub backbuffer: Option<Vec<B::Image>>,
    pub format: hal::format::Format,
    pub extent: hal::image::Extent,
}

impl<B: hal::Backend> Swapchain<B> {
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

impl<B: hal::Backend> Drop for Swapchain<B> {
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
