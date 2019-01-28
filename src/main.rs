#![feature(custom_attribute)]
#![feature(duration_as_u128)]

#[macro_use]
extern crate vulkano_shader_derive;
#[macro_use]
extern crate vulkano_shaders;
extern crate vulkano_win;
#[macro_use]
extern crate vulkano;
extern crate winit;
extern crate image;
extern crate specs;
extern crate uuid;

mod render_layers;
mod events;
mod primitives;
mod utils;
mod renderer;
mod components;
mod timing;

use std::sync::{
    Mutex,
    Arc,
};

use specs::{
    prelude::*,
    world::Builder,
    World,
    DispatcherBuilder
};

use rand::Rng;

use crate::events::event_handler::EventHandler;
use crate::renderer::HelloTriangleApplication;
use crate::primitives::vertex::Vertex;
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::primitives::three_d::cube::Cube;
use crate::components::camera::Camera;

fn main() {
    let mut app = HelloTriangleApplication::initialize();

    let event_handler = Arc::new(Mutex::new(EventHandler::new()));

    let mut world = World::new();
    world.register::<Transform>();
    world.register::<Mesh>();
    world.register::<Camera>();

    let mut dispatcher = DispatcherBuilder::new()
        .build();

    dispatcher.setup(&mut world.res);

    world
        .create_entity()
        .with(Transform::new())
        .with(Camera{ displaying: true })
        .build();

    let mut rng = rand::thread_rng();
    for i in 0..255 {
        let (mut transform, mesh) = Cube::new();
        let x = rng.gen_range(-5.0, 5.0);
        let y = rng.gen_range(-5.0, 5.0);
        let z = rng.gen_range(-5.0, 5.0);
        transform.translate(glm::vec3(x, y, z));

        world
            .create_entity()
            .with(transform)
            .with(mesh)
            .build();
    }

    app.create_scene_vertex_buffers((&world.read_storage::<Transform>(), &world.read_storage::<Mesh>()).join().collect());

    loop {
        // pull in events from windowing system
        app.events_loop.poll_events(|ev| {
            event_handler.lock().unwrap().queue_window_event(ev);
        });

        event_handler.lock().unwrap().process_window_events();
        event_handler.lock().unwrap().handle_events(&world);
        //app.ui_layer.lock().unwrap().push_events_to_widgets(event_handler.application_events.drain(0..).collect());

        // update the systems
        dispatcher.dispatch(&world.res);
        world.maintain();

        // actually render the screen
        // let added_meshes = (world.read_storage::<AddedMesh>(), world.read_storage::<Transform>(), world.read_storage::<Mesh>()).join();
        // let removed_meshes = (world.read_storage::<RemovedMesh>(), world.read_resource::<Transform>(), world.read_storage::<Mesh>()).join();

        // app.scene_layer.lock().unwrap().add_meshes(added_meshes);
        // renderer.remove_meshes(removed_meshes);
        let camera_storage = world.read_storage::<Camera>();
        let transform_storage = world.read_storage::<Transform>();

        let (camera, transform) = (&camera_storage, &transform_storage).join().nth(0).unwrap();

        app.create_command_buffers(transform);
        app.draw_frame();
    }
}
