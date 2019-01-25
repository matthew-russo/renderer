use std::sync::{Arc, Mutex};

use winit::{
    Event,
    WindowEvent,
    DeviceEvent,
    ModifiersState,
    ElementState,
    MouseButton,
    TouchPhase,
    MouseScrollDelta,
    KeyboardInput,
};

use crate::events::application_events::ApplicationEvent;
use crate::events::application_events::KeyPress;
use crate::primitives::two_d::widget::Widget;

// This struct takes all incoming window events and converts them to application events to be passed down to widgets
pub struct EventHandler {
    window_events: Vec<Event>,

    // todo -> not make this pub
    pub application_events: Vec<ApplicationEvent>,

    prev_mouse_position: (f64, f64),
}

impl EventHandler {
    pub fn new() -> EventHandler {
        EventHandler {
            window_events: vec![],
            application_events: vec![],
            prev_mouse_position: (0.0, 0.0)
        }
    }

    fn widget_should_receive_event(event: ApplicationEvent, widget: Arc<Mutex<Widget>>) -> bool {
        // todo -> calculate if this widget should receive the event
        false
    }

    fn transform_event(window_event: Event) -> Vec<ApplicationEvent> {
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
                    WindowEvent::MouseInput { state, button, modifiers, .. } => Self::handle_mouse_input(state, button, modifiers),
                    WindowEvent::TouchpadPressure { .. } => vec![],
                    WindowEvent::AxisMotion { .. } => vec![],
                    WindowEvent::Refresh => vec![],
                    WindowEvent::Touch(_) => vec![],
                    WindowEvent::HiDpiFactorChanged(_) => vec![],
                }
            },
            Event::DeviceEvent { event, .. } => {
                match event {
                    DeviceEvent::Added => vec![],
                    DeviceEvent::Removed => vec![],
                    DeviceEvent::MouseMotion { .. } => vec![],
                    DeviceEvent::MouseWheel { .. } => vec![],
                    DeviceEvent::Motion { .. } => vec![],
                    DeviceEvent::Button { .. } => vec![],
                    DeviceEvent::Key(_) => vec![],
                    DeviceEvent::Text { .. } => vec![],
                }
            },
            Event::Awakened => vec![],
            Event::Suspended(_) => vec![],
        }
    }

    pub fn process_window_events(&mut self) {
        // todo - timing & proper queueing
        self.application_events = self.window_events
            .drain(0..)
            .flat_map(Self::transform_event)
            .collect();
    }

    pub fn queue_window_event(&mut self, window_event: Event) {
        self.window_events.push(window_event);
    }

    fn handle_keyboard_input(input: KeyboardInput) -> Vec<ApplicationEvent> {
        println!("\n");
        println!("keyboard input!");
        println!("\tscancode {:?}", input.scancode);
        println!("\tstate: {:?}", input.state);
        println!("\tvirtual_keycode: {:?}", input.virtual_keycode);
        println!("\tmodifiers: {:?}", input.modifiers);
        println!("\n");

        return match input.virtual_keycode {
            Some(winit::VirtualKeyCode::Escape) => {
                println!("escape key hit -> adding vertices");

                // ui_layer.lock().unwrap().add_geometry(new_vertices, new_indices);
                return if input.state == winit::ElementState::Released {
                    vec![ApplicationEvent::KeyPress(KeyPress::EscKey)]
                } else {
                    vec![]
                }
            },
            _ => vec![],
        }
    }

    fn handle_mouse_scroll(delta: MouseScrollDelta, phase: TouchPhase, modifiers: ModifiersState) -> Vec<ApplicationEvent> {
        println!("\n");
        println!("mouse scroll!");
        println!("\tdelta: {:?}", delta);
        println!("\tphase: {:?}", phase);
        println!("\tmodifiers: {:?}", modifiers);
        println!("\n");

        vec![]
    }

    fn handle_mouse_input(state: ElementState, button: MouseButton, modifiers: ModifiersState) -> Vec<ApplicationEvent> {
        println!("\n");
        println!("mouse input!");
        println!("\tstate {:?}", state);
        println!("\tbutton: {:?}", button);
        println!("\tmodifiers: {:?}", modifiers);
        println!("\n");

        vec![]
    }
}
