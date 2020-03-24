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
extern crate legion;
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
mod xr;

use std::sync::{
    Arc,
    RwLock
};

use rand::Rng;

use cgmath::Vector3;

use crate::renderer::renderer::{Renderer, create_backend};
use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::components::color::Color;
use crate::components::texture::Texture;
use crate::components::camera::Camera;
use crate::components::config::Config;
use crate::primitives::three_d::cube::Cube;
use crate::primitives::drawable::Drawable;
use crate::primitives::uniform_buffer_object::ObjectUniformBufferObject;
use crate::timing::Time;
use crate::systems::rotation::Rotation;
use crate::renderer::renderer::DIMS;
use crate::events::event_handler::EventHandler;
use crate::xr::xr::Xr;

use legion::Universe;
use legion::query::{Read, Write, IntoQuery, Query};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

fn main() {
    env_logger::init();

    let xr = Xr::init();

    let window_builder = winit::window::WindowBuilder::new()
        .with_title("matthew's fabulous rendering engine")
        .with_inner_size(winit::dpi::LogicalSize::new(DIMS.width as f64, DIMS.height as f64));
    let event_loop = winit::event_loop::EventLoop::new();
    let (backend_state, _instance) = create_backend(window_builder, &event_loop);
    let renderer = unsafe { Renderer::new(backend_state) };
    let event_handler = Arc::new(RwLock::new(EventHandler::new()));

    unsafe {
        println!("here");
        let vulkan_xr_session_create_info = renderer.vulkan_session_create_info();
        println!("and there");
    }

    start_engine(renderer, &event_handler);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {
                *control_flow = ControlFlow::Wait;
            },
        }

        let mut new_events = EventHandler::transform_event(event);
        event_handler.write().unwrap().application_events.append(&mut new_events);
    });
}

fn start_engine(mut renderer: Renderer<impl hal::Backend>, event_handler_shared: &Arc<RwLock<EventHandler>>) {
    let event_handler = event_handler_shared.clone();

    std::thread::spawn(move || {
        let time = Arc::new(RwLock::new(Time::new()));
        let rotation_system = Rotation::new(&time);

        // Create a world to store our entities
        // TODO -> create universe with logger
        let universe = Universe::new(None);
        let mut world = universe.create_world();

        world.insert_from(
            (),
            vec![(Transform::new(), Camera { displaying: true })],
        );

        let mut objects = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..64 {
            let (mut transform, mesh) = Cube::new();
            let x = rng.gen_range(-15.0, 15.0);
            let y = rng.gen_range(-15.0, 15.0);
            let z = rng.gen_range(-15.0, 15.0);
            transform.translate(Vector3::new(x, y, z));

            let i: u32 = rng.gen_range(0, 3);

            let texture = Texture {
                path: match i {
                    0 => "textures/container.jpg".into(),
                    1 => "textures/demo.jpg".into(),
                    2 => "textures/wall.jpg".into(),
                    _ => unreachable!()
                }
            };

            objects.push((transform, mesh, texture));
        }

        world.insert_from(
            (),
            objects,
        );

        world.insert_from(
            (),
            vec![(Config::new() ,)],
        );

        {
            let drawables = <(Read<Transform>, Read<Mesh>)>::query()
                .iter_entities(&world)
                .map(|(entity, (transform, mesh))| {
                    let mut drawable = Drawable::new(mesh.clone(), transform.clone());

                    if let Some(color) = world.entity_data::<Color>(entity) {
                        drawable.with_color(color.clone());
                    }

                    if let Some(texture) = world.entity_data::<Texture>(entity) {
                        drawable.with_texture(texture.clone());
                    }

                    drawable
                })
                .collect();

            unsafe {
                renderer.update_drawables(drawables);
            }
        }

        let mut frame_count = 0;
        loop {
            frame_count += 1;
            event_handler.write().unwrap().handle_events(&world);
            rotation_system.run(&world);

            // update frame timing
            time.write().unwrap().tick();

            let uniform_data: Vec<ObjectUniformBufferObject> = <(Read<Transform>, Read<Mesh>)>::query()
                .iter(&mut world)
                .map(|(transform, _mesh)| {
                    transform.clone().to_ubo()
                })
                .collect();

            let camera_transform = <(Read<Transform>, Read<Camera>)>::query()
                .iter(&mut world)
                .map(|(transform, _cam)| transform.clone())
                .next()
                .unwrap();


            let mut need_to_update_config = false;
            if <Read<Config>>::query().iter(&mut world).next().unwrap().should_record_commands {
                let drawables = <(Read<Transform>, Read<Mesh>)>::query()
                    .iter_entities(&world)
                    .map(|(entity, (transform, mesh))| {
                        let mut drawable = Drawable::new(mesh.clone(), transform.clone());

                        if let Some(color) = world.entity_data::<Color>(entity) {
                            drawable.with_color(color.clone());
                        }

                        if let Some(texture) = world.entity_data::<Texture>(entity) {
                            drawable.with_texture(texture.clone());
                        }

                        drawable
                    })
                    .collect();

                unsafe { renderer.update_drawables(drawables) };
                need_to_update_config = true;
            }

            if need_to_update_config {
                let config = <Write<Config>>::query()
                    .iter(&mut world)
                    .next()
                    .unwrap();

                config.should_record_commands = false;
            }

            unsafe { renderer.map_uniform_data(uniform_data) };
            unsafe { renderer.draw_frame(&camera_transform) };
        }
    });
}
