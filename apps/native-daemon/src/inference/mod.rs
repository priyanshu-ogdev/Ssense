pub mod grammar;
pub mod hardware_profiler;
pub mod local_engine;

// Re-export core structs for easier access in main.rs
pub use hardware_profiler::HardwareProfiler;
pub use local_engine::{LocalEngine, ModelType};