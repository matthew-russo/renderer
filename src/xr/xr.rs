use openxr::{ApplicationInfo, Entry, FormFactor, Instance, SystemId};

pub(crate) struct Xr {
    instance: Instance,
    system: SystemId,
}

impl Xr {
    pub fn init() -> Self {
        let entry = Entry::linked();

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

        panic!("welcome to the danger zone");
        // Self {
        //     instance,
        // }
    }
}