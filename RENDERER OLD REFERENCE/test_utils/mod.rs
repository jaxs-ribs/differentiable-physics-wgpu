pub mod scene;
pub mod gpu_helpers;
pub mod simulation_params;
pub mod math;
pub mod benchmark;
pub mod harness;

pub use scene::SceneBuilder;
pub use gpu_helpers::GpuBufferHelpers;
pub use simulation_params::TestSimulationParams;
pub use math::MatrixOperations;
pub use benchmark::{BenchmarkTimer, BenchmarkResults};
pub use harness::TestHarness;

#[cfg(feature = "viz")]
pub use harness::VisualizationTestHarness;