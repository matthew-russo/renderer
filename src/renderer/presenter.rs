use std::sync::{Arc, RwLock};
use crate::renderer::core::GfxBackend;
use hal::window::{Surface, Swapchain};


const DIMS: Extent2D = Extent2D { width: 1024, height: 768 };

type ImageIndex = u32;

trait Presenter<B: hal::Backend> {
    fn acquire_image(&self, acquire_semaphore: B::Semaphore) -> Result<ImageIndex, String>;
    fn present(&self) -> Result<(), String>;
}

struct XrPresenter {

}

struct MonitorPresenter<B: hal::Backend> {
    device: Arc<RwLock<GfxDevice<B>>>,
    swapchain: Swapchain<B>,
    acquired_image: Option<ImageIndex>,
    viewport: Viewport,
}

impl <B: hal::Backend> MonitorPresenter<B> {
    fn new(device: &Arc<RwLock<GfxDevice<B>>>) -> Self {
        let swapchain = SwapchainState::new(
            &mut backend,
            &device);

        let framebuffer = Framebuffer::new(
            &device,
            &mut swapchain,
            &render_pass,
            depth_image_stuff
        );

        let viewport = Self::create_viewport(&swapchain);

        Self {
            device: Arc::clone(device),
            swapchain,
            acquired_image: None,
            viewport,
        }
    }

    fn create_viewport(swapchain_state: &SwapchainState<B>) -> hal::pso::Viewport {
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

    fn present(&mut self) -> Result<(), String> {
        let image_index = self
            .acquired_image
            .take()
            .ok_or(String::from("no image acquired to present to"))?;

        let queue = self
            .device
            .write()
            .unwrap()
            .queue_group.queues[0];

        self.swapchain
            .swapchain
            .unwrap()
            .present(
                queue,
                image_index,
                Some(&*image_present_semaphore)
            )
    }
}

struct Swapchain<B: hal::Backend> {
    swapchain: Option<B::Swapchain>,
    backbuffer: Option<Vec<B::Image>>,
    format: hal::format::Format,
    extent: hal::image::Extent,
    device_state: Arc<RwLock<DeviceState<B>>>
}

impl<B: hal::Backend> Swapchain<B> {
    fn new(backend: &mut GfxBackend<B>, device_state: &Arc<RwLock<GfxDevice<B>>>) -> Self {
        let caps = backend
            .surface
            .capabilities(&device_state.read().unwrap().physical_device);

        let formats = backend
            .surface
            .supported_formats(&device_state.read().unwrap().physical_device);

        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = SwapchainConfig::from_caps(&caps, format, DIMS);

        let extent = swap_config.extent.to_extent();

        let (swapchain, backbuffer) = unsafe {
            device_state
                .write()
                .unwrap()
                .device
                .create_swapchain(&mut backend.surface, swap_config, None)
        }.expect("Can't create swapchain");

        Self {
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            format,
            extent,
            device_state: device_state.clone(),
        }
    }
}

impl<B: hal::Backend> Drop for Swapchain<B> {
    fn drop(&mut self) {
        unsafe {
            self.device_state
                .read()
                .unwrap()
                .device
                .destroy_swapchain(self.swapchain.take().unwrap());
        }
    }
}
