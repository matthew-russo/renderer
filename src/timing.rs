use std::time::Instant;

pub struct Time {
   start_time: Instant,
   last_frame: Instant,
}

impl Time {
   pub fn new() -> Time {
      let now = Instant::now();

      Time {
         start_time: now,
         last_frame: now,
      }
   }

   pub fn delta_time(&self) -> u128 {
      let duration = Instant::now().duration_since(self.start_time);
      duration.as_micros()
   }

   pub fn tick(&mut self) {
      self.last_frame = Instant::now();
   }
}