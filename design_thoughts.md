#Design Doc
this is a log of my ideas, somewhat documentation, and rambling arguments with myself over designs of current aspects

## Goals
- vr native windowing and interface engine
- ergonomic api that is easy to use at a high level but doesn't block access to making unique low level decisions. I don't want this to be another super high level generic engine.

## Scaling to a usable engine
At this point I have a really basic render loop, multiple layers of rendering, a way to arbitrarily add and remove things
from the rendered images, and a way to react to window events. I have them, but they are horrible to work with and are
not at all practical. The current state is clearly a result of me figuring it out as I go.

There are now to main concerns
 - how do i ergonomically work with the things I want to render
 - how do i efficiently render those things
 
 and ideally they are not at all related. While creating things to actually render and manipulate, I don't want to deal
 with worrying about buffers and pipelines and all that. I want to create a thing maybe with a texture, maybe without,
 maybe with some other property, maybe without, and maybe update that thing when i hit a key or provide some input. And when I'm rendering
 I want to not have to deal with all the user bullshit of state and events and input. I just want textures, vertices, indices
 and how they're all tied together which leads to two main things:
 
 - an ECS front-end to facilitate performant world management
 - a rendering backend based on a a render graph (https://github.com/gfx-rs/gfx/wiki/Frame-graphs)

## Concepts 
### RenderLayers
These represent a draw on the window --
    they have their own pipeline, descriptors, vertex buffers, uniform buffers, etc
    
Currently there is a:
- SceneLayer which consists of a 3D scene with cameras and models
- UiLayer which consists of a 2d canvas applied at the front of the screen (will cover all other layers)
    
### Events
We need a way to recognize and flexibly react to windowing events.
This means widgets need to have access to the layer in some aspect in order to:
 - create new elements
 - modify itself
 - modify other elements(?)
 - trigger other events in/for other components (?)
 
The different ways I can think of handling this are:
 - giving the layer (as a special trait object) to each widget/object and having it call functions when it needs. 
 I don't really like this idea because it requires widgets to:
   - be tightly coupled to the definition of the trait
   - deal with a bunch of global memory -> yuck
   - yeah not doing this
 - more events (Commands?) ->  a new event queue of user events.
   - the widget would have a default function to add events to the global queue
     - (todo: look up how to model this queue)
   - the layer would be the sole thing that could pull from the queue (write anywhere, read "once")
   - ordering of the queue isn't hugely important so could have a pretty basic queue appender 
 - query system -> this is essentially the opposite flow of events. 
   - instead of the widget raising an event and appending to a global queue that is read by the layer,
        the layer would loop over its widgets and ask each widget if it needs to perform any actions.
   - each widget would maintain an internal log of commands, and the layer would pull and execute these. this is probably a 
   - able to design it in a manner that would make
 it easy to implement an event queue in the future.
   - the potential issues I see is that we would need to loop over every widget to see if there are commands to execute
        rather than just checking if there are commands in a single queue.
 - a midway between events and query system 
   - the layer maintains a queue and fills that queue by querying its widgets
   - this way not all commands need to be executed at once
   - the logic to add to the queue (querying) could be swapped for more a more efficient/asynchronous approach later if 
        it proves to be an issue
 I like this approach right now and am going to think a bit more on it 
 
 ### Abstraction between the render layer and users
 Render layer still now well defined and mixes a lot of low level 
 
 ### TODO
  - an efficient way of building ubo arrays
  
  
e