use openxr::{ApplicationInfo, Entry, FormFactor, Instance, SystemId};
use std::path::Path;

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

        println!("\nextension set: {:?}\n", extension_set);

        let instance = entry.create_instance(
            &ApplicationInfo {
                application_name: "sxe test app",
                application_version: 0,
                engine_name: "sxe",
                engine_version: 0,
            },
            &extension_set,
        ).unwrap();

        let properties = instance.properties().unwrap();
        println!("properties: {:?}", properties);

        let mut system_id = instance.system(FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();

        // println!("system properties: {:?}", instance.system_properties(system_id));
        println!("vulkan instance extensions: {:?}", instance.vulkan_instance_extensions(system_id));
        println!("vulkan device extensions: {:?}", instance.vulkan_device_extensions(system_id));

        Self {
            instance,
            system_id
        }
    }
}