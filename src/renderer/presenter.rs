use std::sync::{Arc, RwLock};

trait Presenter {
    fn present();
}

struct OpenXrPresenter {

}

struct MonitorPresenter<B: hal::Backend> {
    viewport: Viewport,
    swapchain_state: Swapchain<B>,
    framebuffer_state: Framebuffer<B>,
}

impl <B: hal::Backend> Presenter for MonitorPresenter<B> {
    fn present() {
        unimplemented!()
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
    fn new(backend_state: &mut BackendState<B>, device_state: &Arc<RwLock<DeviceState<B>>>) -> Self {
        let caps = backend_state
            .surface
            .capabilities(&device_state.read().unwrap().physical_device);

        let formats = backend_state
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
                .create_swapchain(&mut backend_state.surface, swap_config, None)
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

struct Framebuffer<B: hal::Backend> {
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

impl<B: hal::Backend> Framebuffer<B> {
    unsafe fn new(
        device_state: &Arc<RwLock<DeviceState<B>>>,
        swapchain_state: &mut Swapchain<B>,
        render_pass_state: &RenderPassState<B>,
        // TODO -> Get rid of / clean up
        depth_image_stuff: (B::Image, B::Memory, B::ImageView)
    ) -> Self
    {
        // TODO -> create depth image for each frame

        let (frame_images, framebuffers) = {
            let extent = hal::image::Extent {
                width: swapchain_state.extent.width as _,
                height: swapchain_state.extent.height as _,
                depth: 1,
            };

            let pairs = swapchain_state
                .backbuffer
                .take()
                .unwrap()
                .into_iter()
                .map(|image| {
                    let image_view = device_state
                        .read()
                        .unwrap()
                        .device
                        .create_image_view(
                            &image,
                            hal::image::ViewKind::D2,
                            swapchain_state.format,
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

impl<B: hal::Backend> Drop for Framebuffer<B> {
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
