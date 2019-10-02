use std::sync::{Arc, RwLock, Mutex};

use winit::event_loop::{ControlFlow, EventLoopWindowTarget};
use winit::event::{
    Event,
    WindowEvent,
    DeviceEvent,
    ModifiersState,
    ElementState,
    MouseButton,
    TouchPhase,
    MouseScrollDelta,
    KeyboardInput,
    VirtualKeyCode,
};

use specs::{
    World,
    Join,
};

use crate::events::application_events::ApplicationEvent;
use crate::events::application_events::KeyPress;
use crate::primitives::two_d::widget::Widget;
use crate::components::camera::Camera;
use crate::components::transform::Transform;

use cgmath::Vector3;

// This struct takes all incoming window events and converts them to application events to be passed down to widgets
pub struct EventHandler {
    window_events: Arc<RwLock<Vec<Event<()>>>>,

    // todo -> not make this pub
    pub application_events: Vec<ApplicationEvent>,

    prev_mouse_position: (f64, f64),
}

impl EventHandler {
    pub fn new() -> EventHandler {
        EventHandler {
            window_events: Arc::new(RwLock::new(vec![])),
            application_events: vec![],
            prev_mouse_position: (0.0, 0.0)
        }
    }

    pub fn read_events_from_event_loop(&mut self)
        -> impl FnMut(Event<()>, &EventLoopWindowTarget<()>, &mut ControlFlow) {

        let window_events = self.window_events.clone();

        return move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {
                    window_events
                        .write()
                        .unwrap()
                        .push(event.clone());
                    *control_flow = ControlFlow::Wait;
                },
            }
        }
    }

    pub fn process_window_events(&mut self) {
        // todo - timing & proper queueing
        self.application_events = self.window_events
            .write()
            .unwrap()
            .drain(0..)
            .flat_map(Self::transform_event)
            .collect();
    }

    pub fn handle_events(&mut self, world: &World) {
        for event in self.application_events.drain(0..) {
            match event {
                ApplicationEvent::KeyPress(key) => match key {
                    KeyPress::EscKey => {},
                    KeyPress::W => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(0.0, 0.0, 1.0));
                        }
                    },
                    KeyPress::A => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(-1.0, 0.0, 0.0));
                        }
                    },
                    KeyPress::S => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(0.0, 0.0, -1.0));
                        }
                    },
                    KeyPress::D => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(1.0, 0.0, 0.0));
                        }
                    },
                    KeyPress::Space => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(0.0, 1.0, 0.0));
                        }
                    },
                    KeyPress::LShift => {
                        for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                            transform.translate(Vector3::new(0.0, -1.0, 0.0));
                        }
                    },
                },
                ApplicationEvent::MouseMotion { x, y} => {
                    for (_camera, transform) in (&world.read_storage::<Camera>(), &mut world.write_storage::<Transform>()).join() {
                        let x_diff = (x * -0.1) as f32;
                        let y_diff = (y * -0.1) as f32;

                        transform.rotate(x_diff, y_diff, 0.0);
                    }
                },
                ApplicationEvent::MouseScroll { delta } => {

                }
            }
        }
    }

    fn widget_should_receive_event(_event: ApplicationEvent, _widget: Arc<Mutex<dyn Widget>>) -> bool {
        // todo -> calculate if this widget should receive the event
        false
    }

    fn transform_event(window_event: Event<()>) -> Vec<ApplicationEvent> {
        return match window_event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::Resized(_) => vec![],
                    WindowEvent::Moved(_) => vec![],
                    WindowEvent::CloseRequested => panic!("matthew's horrible way of handling window exit"),
                    WindowEvent::Destroyed => vec![],
                    WindowEvent::DroppedFile(_) => vec![],
                    WindowEvent::HoveredFile(_) => vec![],
                    WindowEvent::HoveredFileCancelled => vec![],
                    WindowEvent::ReceivedCharacter(_) => vec![],
                    WindowEvent::Focused(_) => vec![],
                    WindowEvent::KeyboardInput { input, .. } => Self::handle_keyboard_input(input),
                    WindowEvent::CursorMoved { .. } => vec![],
                    WindowEvent::CursorEntered { .. } => vec![],
                    WindowEvent::CursorLeft { .. } => vec![],
                    WindowEvent::MouseWheel { delta, phase, modifiers, .. } => Self::handle_mouse_scroll(delta, phase, modifiers),
                    WindowEvent::MouseInput { state, button, modifiers, .. } => Self::handle_mouse_click(state, button, modifiers),
                    WindowEvent::TouchpadPressure { .. } => vec![],
                    WindowEvent::AxisMotion { .. } => vec![],
                    WindowEvent::Touch(_) => vec![],
                    WindowEvent::HiDpiFactorChanged(_) => vec![],
                    WindowEvent::RedrawRequested => vec![]
                }
            },
            Event::DeviceEvent { event, .. } => {
                match event {
                    DeviceEvent::Added => vec![],
                    DeviceEvent::Removed => vec![],
                    DeviceEvent::MouseMotion { delta } => Self::handle_mouse_motion(delta),
                    DeviceEvent::MouseWheel { .. } => vec![],
                    DeviceEvent::Motion { .. } => vec![],
                    DeviceEvent::Button { .. } => vec![],
                    DeviceEvent::Key(_) => vec![],
                    DeviceEvent::Text { .. } => vec![],
                }
            },
            Event::Suspended => vec![],
            Event::UserEvent(_) => vec![],
            Event::NewEvents(_) => vec![],
            Event::EventsCleared => vec![],
            Event::LoopDestroyed => vec![],
            Event::Resumed => vec![]
        }
    }

    fn handle_keyboard_input(input: KeyboardInput) -> Vec<ApplicationEvent> {
        // println!("\n");
        // println!("keyboard input!");
        // println!("\tscancode {:?}", input.scancode);
        // println!("\tstate: {:?}", input.state);
        // println!("\tvirtual_keycode: {:?}", input.virtual_keycode);
        // println!("\tmodifiers: {:?}", input.modifiers);
        // println!("\n");

        return match input.virtual_keycode {
            Some(key_code) => {
                let key_press = match key_code {
                    VirtualKeyCode::A => Some(KeyPress::A),
                    VirtualKeyCode::D => Some(KeyPress::D),
                    VirtualKeyCode::S => Some(KeyPress::S),
                    VirtualKeyCode::W => Some(KeyPress::W),
                    VirtualKeyCode::Escape => Some(KeyPress::EscKey),
                    VirtualKeyCode::Space => Some(KeyPress::Space),
                    VirtualKeyCode::LShift => Some(KeyPress::LShift),
                    _ => None,
                };

                // ui_layer.lock().unwrap().add_geometry(new_vertices, new_indices);
                return if input.state == winit::event::ElementState::Released {
                    match key_press {
                        Some(k) => vec![ApplicationEvent::KeyPress(k)],
                        None => vec![]
                    }
                } else {
                    vec![]
                }
            },
            _ => vec![],
        }
    }

    fn handle_mouse_scroll(delta: MouseScrollDelta, phase: TouchPhase, modifiers: ModifiersState) -> Vec<ApplicationEvent> {
        // println!("\n");
        // println!("mouse scroll!");
        // println!("\tdelta: {:?}", delta);
        // println!("\tphase: {:?}", phase);
        // println!("\tmodifiers: {:?}", modifiers);
        // println!("\n");

        vec![]
    }

    fn handle_mouse_click(state: ElementState, button: MouseButton, modifiers: ModifiersState) -> Vec<ApplicationEvent> {
        // println!("\n");
        // println!("mouse input!");
        // println!("\tstate {:?}", state);
        // println!("\tbutton: {:?}", button);
        // println!("\tmodifiers: {:?}", modifiers);
        // println!("\n");

        vec![]
    }

    fn handle_mouse_motion(delta: (f64, f64)) -> Vec<ApplicationEvent> {
        vec![ApplicationEvent::MouseMotion { x: delta.0, y: delta.1 }]
    }
}

//                              copied from renderer::draw_frame
//
// self.window_state
//     .events_loop
//     .run(|winit_event, _target, _controlFlow| Self::handle_event(winit_event, &mut camera_transform));
//
// pub fn handle_event(self, winit_event: winit::event::Event<UserEvent>, camera_transform: &mut Transform) {
//     let time_readable = self.time.read().unwrap();
//
//     match winit_event {
//         winit::event::Event::WindowEvent { event, .. } => {
//             match event {
//                 // FORWARD
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::W
//                         ),
//                         ..
//                     },
//                     ..
//                 } => {
//                     let forward = camera_transform.forward();
//                     camera_transform.translate(forward * -1.0 * time_readable.delta_time as f32 * 0.01)
//                 },
//
//                 // BACKWARD
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::S
//                         ),
//                         ..
//                     },
//                     ..
//                 } => {
//                     let forward = camera_transform.forward();
//                     camera_transform.translate(forward * time_readable.delta_time as f32 * 0.01)
//                 }
//
//                 // LEFT
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::A
//                         ),
//                         ..
//                     },
//                     ..
//                 } => {
//                     let left = camera_transform.left();
//                     camera_transform.translate(left * time_readable.delta_time as f32 * 0.01)
//                 },
//
//                 // RIGHT
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::D
//                         ),
//                         ..
//                     },
//                     ..
//                 } => {
//                     let right = camera_transform.right();
//                     camera_transform.translate(right * time_readable.delta_time as f32 * 0.01)
//                 },
//
//                 // UP
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::Space
//                         ),
//                         ..
//                     },
//                     ..
//                 } => camera_transform.translate(Vector3::unit_y() * time_readable.delta_time as f32 * 0.01),
//
//                 // DOWN
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(
//                             winit::event::VirtualKeyCode::LShift
//                         ),
//                         ..
//                     },
//                     ..
//                 } => camera_transform.translate(Vector3::unit_y() * -1.0 * time_readable.delta_time as f32 * 0.01),
//
//
//                 winit::event::WindowEvent::KeyboardInput {
//                     input: winit::event::KeyboardInput {
//                         virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
//                         ..
//                     },
//                     ..
//                 }
//                 | winit::event::WindowEvent::CloseRequested => panic!("matthew's bad way of handling exit"),
//                 _ => (),
//
//             }
//         },
//         winit::event::Event::DeviceEvent { event, .. } => {
//             match event {
//                 winit::event::DeviceEvent::MouseMotion { delta } => {
//                     let (mut x, mut y) = delta;
//
//                     x *= -0.1;
//                     y *= -0.1;
//
//                     camera_transform.rotate(x as f32, y as f32, 0.0);
//                 }
//                 _ => (),
//             }
//         }
//         _ => (),
//     };
// }
