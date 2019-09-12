// #![feature(custom_attribute)]
#![cfg_attr(
    not(
        any(
            feature = "vulkan",
            feature = "dx12",
            feature = "metal",
            feature = "gl"
        )
    ),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(
    not(
        any(
            feature = "vulkan",
            feature = "dx12",
            feature = "metal",
            feature = "gl"
        )
    )
)]
extern crate gfx_backend_empty as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

extern crate gfx_hal as hal;

use std::alloc::System;
#[global_allocator]
static ALLOCATOR: System = System;

extern crate glsl_to_spirv;

extern crate winit;
extern crate image;
extern crate specs;
extern crate cgmath;
extern crate uuid;

#[macro_use]
extern crate log;
extern crate env_logger;

mod events;
mod primitives;
mod renderer;
mod components;
mod timing;
mod systems;

use std::sync::{
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

use crate::renderer::renderer::{Renderer, create_backend};
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::components::color::Color;
use crate::components::texture::Texture;
use crate::primitives::three_d::cube::Cube;
use crate::primitives::drawable::Drawable;
use crate::components::camera::Camera;
use crate::timing::Time;
use crate::systems::rotation::Rotation;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use crate::renderer::renderer::DIMS;

fn main() {
    env_logger::init();

    let time = Arc::new(RwLock::new(Time::new()));

    let window_builder = winit::window::WindowBuilder::new()
        .with_title("matthew's fabulous rendering engine")
        .with_inner_size(winit::dpi::LogicalSize::new(DIMS.width as f64, DIMS.height as f64));
    let event_loop = winit::event_loop::EventLoop::new();
    let (backend_state, _instance) = create_backend(window_builder, &event_loop);
    let mut renderer = unsafe { Renderer::new(backend_state) };


    
    std::thread::spawn(move || {
        let mut world = World::new();
        world.register::<Transform>();
        world.register::<Mesh>();
        world.register::<Camera>();
        world.register::<Texture>();
        world.register::<Color>();

        let mut dispatcher = DispatcherBuilder::new()
            .with(Rotation { time: time.clone() }, "rotation", &[])
            .build();

        dispatcher.setup(&mut world.res);

        world
            .create_entity()
            .with(Transform::new())
            .with(Camera { displaying: true })
            .build();

        let mut rng = rand::thread_rng();
        for _i in 0..16 {
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

        unsafe {
            world.maintain();

            let entities = world.entities();
            let transform_storage = world.read_storage::<Transform>();
            let mesh_storage = world.read_storage::<Mesh>();
            let color_storage = world.read_storage::<Color>();
            let texture_storage = world.read_storage::<Texture>();

            let drawables = (&entities, &transform_storage, &mesh_storage)
                .join()
                .map(|(ent, transform, mesh)| {
                    // println!("Processing entity: {:?}", ent);

                    let mut drawable = Drawable::new(mesh.clone(), transform.clone());

                    match color_storage.get(ent) {
                        Some(color) => { drawable.with_color(color.clone()); },
                        _ => ()
                    }

                    match texture_storage.get(ent) {
                        Some(texture) => { drawable.with_texture(texture.clone()); },
                        _ => ()
                    }

                    drawable
                })
                .collect();

            renderer.update_drawables(drawables);
        }

        loop {
            // update the systems
            dispatcher.dispatch(&world.res);
            world.maintain();

            // update frame timing
            time.write().unwrap().tick();

            let _camera_storage = world.read_storage::<Camera>();
            let transform_storage = world.read_storage::<Transform>();
            let mesh_storage = world.read_storage::<Mesh>();

            // TODO -> get rid of clones? probably expensive? need to profile
            let uniform_data = (&transform_storage, &mesh_storage)
                .join()
                .map(|(transform, _mesh)| transform.clone().to_ubo())
                .collect();

            unsafe { renderer.map_object_uniform_data(uniform_data) }
            unsafe { renderer.draw_frame() };
        }
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::UserEvent(event) => println!("user event: {:?}", event),
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,
        _ => *control_flow = ControlFlow::Wait,
    })
}

