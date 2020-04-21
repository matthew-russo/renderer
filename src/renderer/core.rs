use std::sync::{Arc, RwLock};
use std::ops::Deref;
use hal::adapter::{MemoryType, PhysicalDevice};
use hal::Instance;
use hal::queue::QueueFamily;

pub(crate) struct RendererCore<B: hal::Backend> {
    pub backend: GfxBackend<B>,
    pub device: GfxDevice<B>,
}

impl RendererCore<back::Backend> {
    pub fn new(size: winit::dpi::LogicalSize<f64>, event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        unsafe {
            let window_builder = winit::window::WindowBuilder::new()
                .with_title("sxe")
                .with_inner_size(size);
            let (mut backend, _instance) = create_backend(window_builder, event_loop);

            let device = GfxDevice::new(
                backend.adapter.adapter.take().unwrap(),
                backend.surface.read().unwrap().deref()
            );

            Self {
                backend,
                device,
            }
        }
    }
}

pub(crate) struct GfxAdapter<B: hal::Backend> {
    pub adapter: Option<hal::adapter::Adapter<B>>,
    pub memory_types: Vec<MemoryType>,
    pub limits: hal::Limits,
}

impl <B: hal::Backend> GfxAdapter<B> {
    fn new(adapters: &mut Vec<hal::adapter::Adapter<B>>) -> Self {
        match Self::pick_best_adapter(adapters) {
            Some(adapter) => Self::create_adapter_state(adapter),
            None => panic!("Failed to pick an adapter")
        }
    }

    fn pick_best_adapter(adapters: &mut Vec<hal::adapter::Adapter<B>>) -> Option<hal::adapter::Adapter<B>> {
        if adapters.is_empty() {
            return None;
        }

        // TODO -> smarter adapter selection
        return Some(adapters.remove(0));
    }

    fn create_adapter_state(adapter: hal::adapter::Adapter<B>) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        Self {
            adapter: Some(adapter),
            memory_types,
            limits
        }
    }
}


pub(crate) struct GfxDevice<B: hal::Backend> {
    pub device: B::Device,
    pub physical_device: B::PhysicalDevice,
    pub queue_group: hal::queue::QueueGroup<B>,
    pub queue_family_index: Option<u32>,
}

impl <B: hal::Backend> GfxDevice<B> {
    unsafe fn new(adapter: hal::adapter::Adapter<B>, surface: &dyn hal::window::Surface<B>) -> Self {
        let family = adapter
            .queue_families
            .iter()
            .find(|family|
                surface.supports_queue_family(family) && family.queue_type().supports_graphics())
            .unwrap();

        #[cfg(not(feature = "vulkan"))]
            let family_index = None;

        #[cfg(feature = "vulkan")]
            let family_index = {
            let queue_family_any = family as &dyn std::any::Any;
            let back_queue_family: &back::QueueFamily = queue_family_any.downcast_ref().unwrap();
            Some(back_queue_family.index)
        };

        let mut gpu = adapter
            .physical_device
            .open(&[(family, &[1.0])], hal::Features::empty())
            .unwrap();

        Self {
            device: gpu.device,
            physical_device: adapter.physical_device,
            queue_group: gpu.queue_groups.pop().unwrap(),
            queue_family_index: family_index,
        }
    }
}

pub(crate) struct GfxBackend<B: hal::Backend> {
    pub surface: Arc<RwLock<B::Surface>>,
    pub adapter: GfxAdapter<B>,

    #[cfg(any(feature = "vulkan", feature = "dx11", feature = "dx12", feature = "metal"))]
    #[allow(dead_code)]
    pub window: winit::window::Window,
}

impl <B: hal::Backend> GfxBackend<B> {
    pub fn window(&self) -> &winit::window::Window {
        &self.window
    }
}

#[cfg(not(any(feature="gl", feature="dx12", feature="vulkan", feature="metal")))]
fn create_backend<B: hal::Backend>(window_builder: winit::window::WindowBuilder, event_loop: &winit::event_loop::EventLoop<()>) -> (GfxBackend<back::Backend>, ()) {
    panic!("You must specify one of the valid backends using --features=<backend>, with \"gl\", \"dx12\", \"vulkan\", and \"metal\" being valid backends.");
}

#[cfg(feature="gl")]
fn create_backend(window_builder: winit::window::WindowBuilder, event_loop: &winit::event_loop::EventLoop<()>) -> (GfxBackend<back::Backend>, ()) {
    let (mut adapters, mut surface) = {
        let window = {
            let builder = back::config_context(back::glutin::ContextBuilder::new(), Rgba8Srgb::SELF, None).with_vsync(true);
            back::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
        };

        let surface = back::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (apaters, surface)
    };

    let backend_state = GfxBackend {
        surface: Arc::new(RwLock::new(surface)),
        adapter: GfxAdapter::new(adapters),
    };

    (backend_state, ())
}

#[cfg(any(feature="dx12", feature="vulkan", feature="metal"))]
fn create_backend(window_builder: winit::window::WindowBuilder, event_loop: &winit::event_loop::EventLoop<()>) -> (GfxBackend<back::Backend>, back::Instance) {
    let window = window_builder
        .build(event_loop)
        .unwrap();

    let instance = back::Instance::create("matthew's spectacular rendering engine", 1).expect("failed to create an instance");
    let surface = unsafe { instance.create_surface(&window).expect("Failed to create a surface") };
    let mut adapters = instance.enumerate_adapters();

    let backend_state = GfxBackend {
        surface: Arc::new(RwLock::new(surface)),
        adapter: GfxAdapter::new(&mut adapters),
        window
    };

    (backend_state, instance)
}
