pub unsafe fn any_as_u8_slice<T: Sized>(p: &T, pad_to_size: usize) -> Vec<u8> {
    let mut raw_bytes = std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        std::mem::size_of::<T>())
        .to_vec();

    raw_bytes.resize(pad_to_size, 0);

    raw_bytes
}

pub fn data_path(specific_file: &str) -> PathBuf {
    let root_data = Path::new("src/data");
    let specific_file_path = Path::new(specific_file);
    return root_data.join(specific_file_path);
}
