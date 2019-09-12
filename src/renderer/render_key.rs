#[derive(PartialEq, Eq, Hash)]
pub struct RenderKey {
    pub module: ModuleName,
    pub resource_type: ResourceType,
}

impl RenderKey {
    pub fn new(module: ModuleName, resource_type: ResourceType) -> Self {
        Self {
            module,
            resource_type,
        }
    }
}

impl std::fmt::Display for RenderKey {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}::{}", self.module, self.resource_type)
    }
}

impl From<&Option<crate::components::texture::Texture>> for RenderKey {
    fn from(texture: &Option<crate::components::texture::Texture>) -> Self {
        let tex_path = match texture {
            Some(tex) => tex.path.clone(),
            None => String::from("NULL_TEX"),
        };

        RenderKey::new(ModuleName::BackendRenderer, ResourceType::Texture(tex_path))
    }
}

impl From<&crate::components::texture::Texture> for RenderKey {
    fn from(texture: &crate::components::texture::Texture) -> Self {
        RenderKey::new(ModuleName::BackendRenderer, ResourceType::Texture(texture.path.clone()))
    }
}

#[derive(PartialEq, Eq, Hash)]
pub enum ModuleName {
    BackendRenderer,
}

impl std::fmt::Display for ModuleName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let to_print = match self {
            ModuleName::BackendRenderer => "backend_renderer".to_string(),
            _ => panic!("unknown resource type"),
        };

        write!(f, "{}", to_print)
    }
}

#[derive(Eq)]
pub enum ResourceType {
    Texture(String),
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let to_print = match self {
            ResourceType::Texture(path) => format!("texture-{}", path),
            _ => panic!("unknown resource type"),
        };

        write!(f, "{}", to_print)
    }
}

impl std::cmp::PartialEq for ResourceType {
    fn eq(&self, other: &Self) -> bool {
        let rt1 = format!("{}", self);
        let rt2 = format!("{}", other);

        return rt1 == rt2;
    }
}

impl std::hash::Hash for ResourceType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let formatted = format!("{}", self);
        formatted.hash(state);
    }
}
