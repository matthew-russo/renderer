use std::path::Path;
use ash::vk;
use openxr::{
    ApplicationInfo,
    Entry,
    FormFactor,
    FrameWaiter,
    FrameStream,
    Instance,
    SystemId,
    Session,
    Vulkan,
    Result as OpenXrResult,
    vulkan::SessionCreateInfo,
};

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

    pub unsafe fn create_vulkan_session(&self, session_create_info: VulkanXrSessionCreateInfo) -> OpenXrResult<VulkanXrSession> {
        panic!("not ready yet");
        // let create_info = SessionCreateInfo {
        //     instance: session_create_info.instance,
        //     physical_device: session_create_info.physical_device,
        //     device: session_create_info.device,
        //     queue_family_index: session_create_info.queue_family_index,
        //     queue_index: session_create_info.queue_index,
        // };

        // self.instance
        //     .create_session(self.system_id, &create_info)
        //     .map( VulkanXrSession::from)
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
    session: Session<Vulkan>,
    frame_waiter: FrameWaiter,
    frame_stream: FrameStream<Vulkan>,
}

impl From<(Session<Vulkan>, FrameWaiter, FrameStream<Vulkan>)> for VulkanXrSession {
    fn from(i: (Session<Vulkan>, FrameWaiter, FrameStream<Vulkan>)) -> Self {
        Self {
            session: i.0,
            frame_waiter: i.1,
            frame_stream: i.2,
        }
    }
}