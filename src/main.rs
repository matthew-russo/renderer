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

extern crate log;
extern crate env_logger;

mod events;
mod primitives;
mod renderer;
mod components;
mod timing;
mod systems;
mod utils;

use std::sync::{
    Arc,
    RwLock
};

use rand::Rng;

use cgmath::Vector3;

use crate::components::mesh::Mesh;
use crate::components::transform::Transform;
use crate::components::color::Color;
use crate::components::texture::Texture;
use crate::components::camera::Camera;
use crate::components::config::Config;
use crate::primitives::three_d::cube::Cube;
use crate::primitives::drawable::Drawable;
use crate::timing::Time;
use crate::systems::rotation::Rotation;
use crate::events::event_handler::EventHandler;
// use crate::xr::xr::Xr;

use legion::Universe;
use legion::query::{Read, Write, IntoQuery, Query};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use crate::renderer::core::RendererCore;
use crate::renderer::allocator::{Allocator, GfxAllocator};
use crate::renderer::drawer::{Drawer, GfxDrawer};
use crate::renderer::presenter::{Presenter, MonitorPresenter, XrPresenter};
use crate::primitives::uniform_buffer_object::ObjectUniformBufferObject;

fn main() {
    env_logger::init();

    let default_logical_size = winit::dpi::LogicalSize {
        width: 800.0 as f64,
        height: 600.0 as f64,
    };

    let event_loop = winit::event_loop::EventLoop::new();
    let renderer_core = Arc::new(RwLock::new(RendererCore::new(default_logical_size, &event_loop)));
    let allocator = GfxAllocator::new(&renderer_core);

    #[cfg(feature = "xr")]
    let presenter = XrPresenter::new(&renderer_core, allocator);

    #[cfg(not(feature = "xr"))]
    let presenter = MonitorPresenter::new(&renderer_core, allocator);

    let drawer = GfxDrawer::new(&renderer_core, allocator, presenter.viewport());

    let event_handler = Arc::new(RwLock::new(EventHandler::new()));

    start_engine(drawer, presenter, &event_handler);

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

fn start_engine<B: hal::Backend, D: Drawer<B>, P: Presenter<B>>(mut drawer: D, mut presenter: P, event_handler_shared: &Arc<RwLock<EventHandler>>) {
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
        world.insert_from(
            (),
            generate_n_objs(64),
        );
        world.insert_from(
            (),
            vec![(Config::new() ,)],
        );

        unsafe {
            drawer.update_drawables(fetch_drawables(&world));
        }

        loop {
            event_handler.write().unwrap().handle_events(&world);

            // TODO -> run all systems
            rotation_system.run(&world);

            // update frame timing
            time.write().unwrap().tick();

            let mut need_to_update_config = false;
            if <Read<Config>>::query().iter(&mut world).next().unwrap().should_record_commands {
                unsafe { drawer.update_drawables(fetch_drawables(&world)) };
                need_to_update_config = true;
            }

            if need_to_update_config {
                let config = <Write<Config>>::query()
                    .iter(&mut world)
                    .next()
                    .unwrap();

                config.should_record_commands = false;
            }

            unsafe {
                drawer.update_uniforms(fetch_uniforms(&world))?;
                drawer.update_camera(fetch_camera_transform(&world));
                let image_index = presenter.acquire_image()?;
                drawer.draw(image_index as usize);
                presenter.present();
            }
        }
    });
}

fn generate_n_objs(n: u32) -> Vec<(Transform, Mesh, Texture)> {
    let mut objects = Vec::new();
    let mut rng = rand::thread_rng();

    for _i in 0..n {
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

    objects
}

fn fetch_camera_transform(world: &legion::World) -> Transform {
    <(Read<Transform>, Read<Camera>)>::query()
        .iter(&world)
        .map(|(transform, _cam)| transform.clone())
        .next()
        .unwrap()
}

fn fetch_uniforms(world: &legion::World) -> Vec<ObjectUniformBufferObject> {
    <(Read<Transform>, Read<Mesh>)>::query()
        .iter(world)
        .map(|(transform, _mesh)| {
            transform.clone().to_ubo()
        })
        .collect()
}

fn fetch_drawables(world: &legion::World) -> Vec<Drawable> {
    <(Read<Transform>, Read<Mesh>)>::query()
        .iter_entities(world)
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
        .collect()
}
