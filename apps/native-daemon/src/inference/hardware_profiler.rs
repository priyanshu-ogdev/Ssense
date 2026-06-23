// apps/native-daemon/src/inference/hardware_profiler.rs

use anyhow::{bail, Result};
use sysinfo::{CpuRefreshKind, Disks, MemoryRefreshKind, RefreshKind, System};
use std::thread::available_parallelism; // 🚀 FIX 1: `::` instead of `:`
use tracing::{info, warn};

// Qwen 9B Q4_K_M (5.5GB) requirements + OS buffer
const MIN_REQUIRED_RAM_MB: u64 = 7168; 
const MIN_REQUIRED_DISK_MB: u64 = 6144; // 🚀 FIX 2: `u64` instead of `u62`

/// 🚀 FIX 5: The Telemetry Bridge to LocalEngine
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub optimal_threads: i32,
}

pub struct HardwareProfiler;

impl HardwareProfiler {
    /// Validates the host system's capabilities.
    /// Returns `Ok(HardwareProfile)` containing dynamic thread telemetry.
    pub fn verify_system_capabilities() -> Result<HardwareProfile> {
        info!("🚀 Running hardware capability check...");

        // 🚀 SOTA FIX 1: The 32-bit Virtual Memory Trap
        // mmap-ing a 5.5GB model is mathematically impossible on a 32-bit OS.
        if std::mem::size_of::<usize>() < 8 { // 🚀 FIX 3: `::` instead of `:`
            bail!("Fatal: 32-bit architecture detected. Ssense requires a 64-bit OS to map the 5.5GB neural network into virtual memory.");
        }

        // 🚀 SOTA FIX 2: Zero-Warning Initialization using `nothing()`
        let refresh_kind = RefreshKind::nothing()
            .with_memory(MemoryRefreshKind::everything()) // 🚀 FIX 4: `::` instead of `:`
            .with_cpu(CpuRefreshKind::everything());
            
        let sys = System::new_with_specifics(refresh_kind);

        // 1. MEMORY VALIDATION
        let total_ram_mb = sys.total_memory() / 1024 / 1024;
        let available_ram_mb = sys.available_memory() / 1024 / 1024;

        info!("💾 Memory: {} MB total, {} MB available", total_ram_mb, available_ram_mb);

        #[cfg(target_os = "linux")]
        {
            if let Some(cgroup_limit_mb) = get_cgroup_memory_limit_mb() {
                info!("📦 Container detected. Cgroup memory limit: {} MB", cgroup_limit_mb);
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
            bail!(
                "Insufficient RAM. Ssense requires {} MB free, but only {} MB available.",
                MIN_REQUIRED_RAM_MB,
                available_ram_mb
            );
        }

        // 2. DISK PARTITION VALIDATION
        let data_dir = directories::ProjectDirs::from("com", "Ssense", "SsenseDaemon") // 🚀 FIX 5: `::` instead of `:`
            .map(|p| p.data_local_dir().to_path_buf()) // 🚀 FIX 6: Added missing `)`
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default()); // 🚀 FIX 7: `::` instead of `:`

        let disks = Disks::new_with_refreshed_list(); // 🚀 FIX 8: `::` instead of `:`
        let mut best_match: Option<&sysinfo::Disk> = None;
        let mut max_len = 0;

        // Longest Prefix Match guarantees we check the exact physical partition
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
                    "Insufficient disk space. Ssense requires {} MB free, but only {} MB available on {:?}.",
                    MIN_REQUIRED_DISK_MB,
                    available_mb,
                    disk.mount_point()
                );
            }
        } else {
            warn!("⚠️ Could not isolate the exact disk partition for {:?}. Bypassing strict disk check.", data_dir);
        }

        // 3. CPU PROFILE & DYNAMIC THREAD SCALING
        let logical_cores = sys.cpus().len();
        let physical_cores = sys.physical_core_count().unwrap_or((logical_cores / 2).max(1));
        let cgroup_cores = available_parallelism().map(|n| n.get()).unwrap_or(logical_cores);
        
        // Take the strictest bottleneck (Silicon vs Container).
        let mut optimal_threads = physical_cores.min(cgroup_cores) as i32;

        // Memory bandwidth bottlenecks standard GPUs/CPUs past 8-10 cores for LLM inference.
        if optimal_threads > 8 {
            optimal_threads = 8;
        } else if optimal_threads < 1 {
            optimal_threads = 1;
        }

        info!("🖥️ CPU Profile: Logical: {} | Physical: {} | Cgroup: {} -> Bound LLM to {} threads", 
            logical_cores, physical_cores, cgroup_cores, optimal_threads);

        info!("✅ Hardware verification passed. Ssense is cleared for ignition.");
        
        // 🚀 FIX 9: Return the telemetry to main.rs
        Ok(HardwareProfile { optimal_threads })
    }

    /// Quick check for UI display (non-blocking, returns status)
    pub fn get_system_status() -> SystemStatus {
        let refresh_kind = RefreshKind::nothing().with_memory(MemoryRefreshKind::everything());
        let sys = System::new_with_specifics(refresh_kind);

        let available_ram_mb = sys.available_memory() / 1024 / 1024;
        
        SystemStatus {
            can_load_model: available_ram_mb >= MIN_REQUIRED_RAM_MB,
            available_ram_mb,
            required_ram_mb: MIN_REQUIRED_RAM_MB, // 🚀 FIX 10: `u64` instead of `i64`
        }
    }
}

#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub can_load_model: bool,
    pub available_ram_mb: u64,
    pub required_ram_mb: u64,
}

#[cfg(target_os = "linux")]
fn get_cgroup_memory_limit_mb() -> Option<u64> {
    // Try cgroup v2 first (Standard on modern Linux/WSL2)
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory.max") {
        if content.trim() != "max" {
            if let Ok(bytes) = content.trim().parse::<u64>() {
                return Some(bytes / 1024 / 1024);
            }
        }
    }
    // Fallback to cgroup v1 (Older Docker configurations)
    if let Ok(content) = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(bytes) = content.trim().parse::<u64>() {
            // v1 often returns a massive number (like 9223372036854771712) to indicate "no limit"
            if bytes < 100_000_000_000 { 
                return Some(bytes / 1024 / 1024);
            }
        }
    }
    None
}