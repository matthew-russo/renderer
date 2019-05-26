use std::time::Instant;

pub struct Time {
   start_time: Instant,
   last_frame: Instant,
   pub delta_time: u128,
}

impl Time {
   pub fn new() -> Time {
      let now = Instant::now();

      Time {
         start_time: now,
         last_frame: now,
         delta_time: 0,
      }
   }

   pub fn total_time(&self) -> u128 {
      let duration = Instant::now().duration_since(self.start_time);
      duration.as_millis()
   }

   pub fn tick(&mut self) {
      let now = Instant::now();
      self.delta_time = now
          .duration_since(self.last_frame)
          .as_millis();
      self.last_frame = now;
   }
}
