use anyhow::{bail, Result};
use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};
use std::thread::available_parallelism; 
use tracing::{info, warn};

// SOTA FIX: Qwen 2.5 7B Q4_K_M is exactly ~4.3GB. 
// We require 4300MB for the LLM + ~300MB for ONNX RAG + ~500MB OS overhead = 5100MB
const MIN_REQUIRED_RAM_MB: u64 = 5100; 

// The total size of the 2 models + safetensors + ONNX downloaded locally is ~9.5GB
const MIN_REQUIRED_DISK_MB: u64 = 10000; 

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub optimal_threads: i32,
    pub has_gpu_acceleration: bool,
}

pub struct HardwareProfiler;

impl HardwareProfiler {
    /// Validates the host system's capabilities.
    /// Returns `Ok(HardwareProfile)` containing dynamic thread telemetry.
    pub fn verify_system_capabilities() -> Result<HardwareProfile> {
        info!("🚀 [Profiler] Running Edge hardware capability check...");

        // 1. ARCHITECTURE VALIDATION
        if std::mem::size_of::<usize>() < 8 { 
            bail!("Fatal: 32-bit architecture detected. Ssense requires a 64-bit OS to map the neural network into virtual memory.");
        }

        let refresh_kind = RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::everything()) 
            .with_cpu(CpuRefreshKind::everything());
            
        let sys = System::new_with_specifics(refresh_kind);

        // 2. MEMORY VALIDATION
        let total_ram_mb = sys.total_memory() / 1024 / 1024;
        let available_ram_mb = sys.available_memory() / 1024 / 1024;

        info!("💾 [Profiler] System Memory: {} MB total, {} MB currently available", total_ram_mb, available_ram_mb);

        #[cfg(target_os = "linux")]
        {
            if let Some(cgroup_limit_mb) = get_cgroup_memory_limit_mb() {
                info!("📦 [Profiler] Container detected. Cgroup memory limit: {} MB", cgroup_limit_mb);
                if cgroup_limit_mb < MIN_REQUIRED_RAM_MB {
                    bail!(
                        "Insufficient container memory. Ssense requires {} MB, but cgroup limit is {} MB.",
                        MIN_REQUIRED_RAM_MB,
                        cgroup_limit_mb
                    );
                }
            }
        }

        if available_ram_mb < MIN_REQUIRED_RAM_MB {
            warn!(
                "⚠️ [Profiler] Low RAM Warning: Ssense requires {} MB free, but only {} MB available. OS paging (Swap) may cause severe inference latency.",
                MIN_REQUIRED_RAM_MB,
                available_ram_mb
            );
            // SOTA FIX: We warn instead of bail. If they have 8GB total, they might have exactly 4.8GB available.
            // mmap will handle the paging, it will just be slow. Let the user decide.
        }

        // 3. DISK PARTITION VALIDATION
        let data_dir = directories::ProjectDirs::from("com", "Ssense", "ssense-native-daemon") 
            .map(|p| p.data_dir().to_path_buf()) 
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default()); 

        let disks = Disks::new_with_refreshed_list(); 
        let mut best_match: Option<&sysinfo::Disk> = None;
        let mut max_len = 0;

        for disk in disks.iter() {
            let mount_point = disk.mount_point();
            if data_dir.starts_with(mount_point) {
                let len = mount_point.as_os_str().len();
                if len > max_len {
                    max_len = len;
                    best_match = Some(disk);
                }
            }
        }

        if let Some(disk) = best_match {
            let available_mb = disk.available_space() / 1024 / 1024;
            if available_mb < MIN_REQUIRED_DISK_MB {
                bail!(
                    "Insufficient disk space. Ssense Offline Mode requires {} MB free, but only {} MB available on {:?}.",
                    MIN_REQUIRED_DISK_MB,
                    available_mb,
                    disk.mount_point()
                );
            }
        } else {
            warn!("⚠️ [Profiler] Could not isolate the exact disk partition for {:?}. Bypassing strict disk check.", data_dir);
        }

        // 4. CPU PROFILE & DYNAMIC THREAD SCALING
        let logical_cores = sys.cpus().len();
        let physical_cores = sys.physical_core_count().unwrap_or((logical_cores / 2).max(1));
        let cgroup_cores = available_parallelism().map(|n| n.get()).unwrap_or(logical_cores);
        
        // Take the strictest bottleneck (Silicon vs Container).
        let mut optimal_threads = physical_cores.min(cgroup_cores) as i32;

        // LLM memory bandwidth bottlenecks standard CPUs past 8-10 cores.
        if optimal_threads > 8 {
            optimal_threads = 8;
        } else if optimal_threads < 1 {
            optimal_threads = 1;
        }

        let has_gpu_acceleration = cfg!(feature = "cublas") || cfg!(feature = "metal");

        info!("🖥️ [Profiler] CPU Profile: Logical: {} | Physical: {} | Cgroup: {} -> Bound LLM to {} threads", 
            logical_cores, physical_cores, cgroup_cores, optimal_threads);

        info!("✅ [Profiler] Hardware verification passed. Ssense is cleared for ignition.");
        
        Ok(HardwareProfile { optimal_threads, has_gpu_acceleration })
    }

    /// Quick check for UI display (non-blocking, returns status)
    pub fn get_system_status() -> SystemStatus {
        let refresh_kind = RefreshKind::nothing().with_memory(MemoryRefreshKind::everything());
        let sys = System::new_with_specifics(refresh_kind);

        let available_ram_mb = sys.available_memory() / 1024 / 1024;
        let has_gpu = cfg!(feature = "cublas") || cfg!(feature = "metal");
        
        SystemStatus {
            can_load_model: available_ram_mb >= MIN_REQUIRED_RAM_MB,
            available_ram_mb,
            required_ram_mb: MIN_REQUIRED_RAM_MB, 
            has_gpu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub can_load_model: bool,
    pub available_ram_mb: u64,
    pub required_ram_mb: u64,
    pub has_gpu: bool,
}

#[cfg(target_os = "linux")]
fn get_cgroup_memory_limit_mb() -> Option<u64> {
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        if content.trim() != "max" {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                return Some(bytes / 1024 / 1024);
            }
        }
    }
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(bytes) = content.trim().parse::<u64>() {
            if bytes < 100_000_000_000 { 
                return Some(bytes / 1024 / 1024);
            }
        }
    }
    None
}