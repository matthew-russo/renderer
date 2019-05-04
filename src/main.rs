#![feature(custom_attribute)]
#![feature(duration_as_u128)]
#![feature(copysign)]

#![cfg_attr(
    not(any(
        feature = "vulkan",
        feature = "dx12",
        feature = "metal",
        feature = "gl"
    )),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;

extern crate glsl_to_spirv;

extern crate winit;
extern crate image;
extern crate specs;
extern crate cgmath;
extern crate uuid;


mod events;
mod primitives;
mod utils;
mod renderer;
mod components;
mod timing;
mod systems;

use std::sync::{
    Mutex,
    Arc,
    RwLock
};

use specs::{
    prelude::*,
    world::Builder,
    World,
    DispatcherBuilder
};

use rand::Rng;

use cgmath::Vector3;

use crate::events::event_handler::EventHandler;
use crate::renderer::Renderer;
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::primitives::three_d::cube::Cube;
use crate::components::camera::Camera;
use crate::timing::Time;
use crate::systems::rotation::Rotation;

fn main() {
    let mut time = Arc::new(RwLock::new(Time::new()));
    let mut renderer = Renderer::initialize();

    let event_handler = Arc::new(Mutex::new(EventHandler::new()));

    let mut world = World::new();
    world.register::<Transform>();
    world.register::<Mesh>();
    world.register::<Camera>();

    let mut dispatcher = DispatcherBuilder::new()
        .with(Rotation { time: time.clone() }, "rotation", &[])
        .build();

    dispatcher.setup(&mut world.res);

    world
        .create_entity()
        .with(Transform::new())
        .with(Camera{ displaying: true })
        .build();

    let mut rng = rand::thread_rng();
    for i in 0..512 {
        let (mut transform, mesh) = Cube::new();
        let x = rng.gen_range(-15.0, 15.0);
        let y = rng.gen_range(-15.0, 15.0);
        let z = rng.gen_range(-15.0, 15.0);
        transform.translate(Vector3::new(x, y, z));

        world
            .create_entity()
            .with(transform)
            .with(mesh)
            .build();
    }

    loop {
        // pull in events from windowing system
        // renderer.events_loop.poll_events(|ev| {
        //     event_handler.lock().unwrap().queue_window_event(ev);
        // });

        // event_handler.lock().unwrap().process_window_events();
        // event_handler.lock().unwrap().handle_events(&world);

        // update the systems
        dispatcher.dispatch(&world.res);
        world.maintain();

        // update frame timing
        time.write().unwrap().tick();

        let camera_storage = world.read_storage::<Camera>();
        let transform_storage = world.read_storage::<Transform>();
        let mesh_storage = world.read_storage::<Mesh>();

        let (camera, transform) = (&camera_storage, &transform_storage).join().nth(0).unwrap();

        let renderables = (&transform_storage, &mesh_storage)
            .join()
            .map(|(transform, mesh)| (mesh.key.clone(), transform.clone()))
            .collect();

        // renderer.create_command_buffers(transform, renderables);
        renderer.draw_frame(&mut event_handler);
    }
}
