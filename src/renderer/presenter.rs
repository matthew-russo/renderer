use std::sync::{Arc, RwLock};
use hal::window::{Extent2D};
use hal::pso::Viewport;
use hal::device::Device;
use hal::window::Surface;
use crate::renderer::core::RendererCore;

const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };

type ImageIndex = u32;

pub(crate) trait Presenter<B: hal::Backend> {
    fn acquire_image(&mut self, acquire_semaphore: B::Semaphore) -> Result<ImageIndex, String>;
    fn present(&mut self) -> Result<(), String>;
    fn viewport(&self) -> Viewport;
}

pub(crate) struct XrPresenter {

}

pub(crate) struct MonitorPresenter<B: hal::Backend> {
    core: Arc<RwLock<RendererCore<B>>>,
    swapchain: Swapchain<B>,
    acquired_image: Option<ImageIndex>,
    viewport: Viewport,
}

impl <B: hal::Backend> MonitorPresenter<B> {
    pub fn new(core: &Arc<RwLock<RendererCore<B>>>) -> Self {
        let swapchain = Swapchain::new(core);
        let viewport = Self::create_viewport(&swapchain);
        Self {
            core: Arc::clone(core),
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

impl <B: hal::Backend> Presenter<B> for MonitorPresenter<B> {
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
