use crate::primitives::vertex::Vertex;
use crate::components::texture::Texture;

#[derive(Clone, Debug)]
pub struct UiDrawData {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub commands: Vec<UiDrawCommand>,
}

impl From<&imgui::DrawData> for UiDrawData {
    fn from(imgui_draw_data: &imgui::DrawData) -> Self {
        let mut indices = vec![];
        let mut vertices = vec![];
        let mut commands = vec![];

        let mut index_offset = 0;
        let mut vertex_offset = 0;
        for draw_list in imgui_draw_data.draw_lists() {
            let mut new_indices: Vec<u32> = draw_list
                .idx_buffer()
                .iter()
                .map(|i| (*i as u32) + index_offset)
                .collect();
            indices.append(&mut new_indices);

            let mut new_vertices: Vec<Vertex> = draw_list
                .vtx_buffer()
                .iter()
                .map(Vertex::from)
                .collect();
            vertices.append(&mut new_vertices);

            let mut new_commands: Vec<UiDrawCommand> = draw_list
                .commands()
                .map(|dc| UiDrawCommand::from((dc, (index_offset, vertex_offset))))
                .collect();
            commands.append(&mut new_commands);

            index_offset += indices.len() as u32;
            vertex_offset += vertices.len() as u32;
        }

        Self {
            indices,
            vertices,
            commands,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UiDrawCommand {
    pub count: u32,
    pub clip_rect: [f32; 4],
    pub texture_id: Option<Texture>,
    pub vtx_offset: u32,
    pub idx_offset: u32,
}

impl From<(imgui::DrawCmd, (u32, u32))> for UiDrawCommand {
    fn from(draw_cmd_and_offsets: (imgui::DrawCmd, (u32, u32))) -> Self {
        let draw_cmd = draw_cmd_and_offsets.0;
        let index_and_vertex_offsets = draw_cmd_and_offsets.1;
        let index_offset = index_and_vertex_offsets.0;
        let vertex_offset = index_and_vertex_offsets.1;

        if let imgui::DrawCmd::Elements { count, cmd_params} = draw_cmd {
            Self {
                count: count as u32,
                clip_rect: cmd_params.clip_rect,
                texture_id: None, //cmd_params.texture_id,
                vtx_offset: cmd_params.vtx_offset as u32 + vertex_offset,
                idx_offset: cmd_params.idx_offset as u32 + index_offset,
            }
        } else {
            unimplemented!("Don't know what to do with other imgui::DrawCmd variants")
        }
    }
}
