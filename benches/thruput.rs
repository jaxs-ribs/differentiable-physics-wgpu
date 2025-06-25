use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_physics_step(c: &mut Criterion) {
    c.bench_function("physics_step_1k", |b| {
        b.iter(|| {
            // TODO: Implement actual physics step
            black_box(42);
        });
    });
}

criterion_group!(benches, benchmark_physics_step);
criterion_main!(benches);