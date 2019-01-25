use crate::primitives::vertex::Vertex;
use crate::events::application_events::{ApplicationEvent, KeyPress};
use crate::primitives::two_d::quad::Quad;

pub trait Widget {
    //fn register_events(events: HashMap<String, ApplicationEvent>);
    //fn commands(&mut self) -> Option<Vec<Command>>;
    fn on(&mut self, event: &ApplicationEvent) -> bool;
    fn quad(&self) -> Quad;
}

pub struct EscMenu {
    quad: Quad,
}

impl EscMenu {
    pub fn new() -> EscMenu {
        let mut main_box = Quad::new("escmenu".to_string(), 0.5, -0.5, 0.5, 0.5, None);
        let child_box1 = Quad::new("escmenu_child1".to_string(), 0.7, -0.85, 0.3, 0.85, Some(&main_box));
        let child_box2 = Quad::new("escmenu_child2".to_string(), 0.0, -0.5, -0.9, 0.3, Some(&main_box));

        main_box.rendered = false;

        EscMenu {
            quad: main_box.with_children(vec![child_box1, child_box2]),
        }
    }
}

impl Widget for EscMenu {
    // static and will be much quicker. we just pull all registrations for <handle> and call them`
    // fn register_events(events: HashMap<String, ApplicationEvent>) -> HashMap<ApplicationEventHandle, ApplicationEventRegistration> {
    //     let escKeyHandler = |e| => {

    //     }
    // }

    // raw -- we loop all widgets and call on with every event. leaving this here just as a reminder since this is essentially the function that is type ApplicationEventRegistration
    fn on(&mut self, event: &ApplicationEvent) -> bool {
        match event {
            ApplicationEvent::KeyPress(KeyPress::EscKey) => {
                // registrations.push(KeyboardInput::Esckey);
                self.quad.rendered = !self.quad.rendered;
                return true;
            },
            _ => {
                return false;
            },
        }
    }

    fn quad(&self) -> Quad {
        self.quad.clone()
    }

    // fn commands(&mut self) -> Option<Vec<Command>> {
    //     match self.commands.len() {
    //         0 => None,
    //         _ => Some(self.commands.drain().collect())
    //     }
    // }
}