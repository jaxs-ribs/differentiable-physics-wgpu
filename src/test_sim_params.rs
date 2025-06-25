use std::mem;

#[repr(C)]
struct SimParams {
    dt: f32,
    gravity: [f32; 3],
    num_bodies: u32,
    _padding: [f32; 3],
}

fn main() {
    println\!("Size of SimParams: {} bytes", mem::size_of::<SimParams>());
    println\!("Offset of dt: {}", mem::offset_of\!(SimParams, dt));
    println\!("Offset of gravity: {}", mem::offset_of\!(SimParams, gravity));
    println\!("Offset of num_bodies: {}", mem::offset_of\!(SimParams, num_bodies));
}
EOF < /dev/null