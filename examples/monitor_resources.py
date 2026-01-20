"""Resource monitoring for local CPU/memory and remote GPU."""
import asyncio
import subprocess
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_util: float | None = None
    gpu_memory_mb: float | None = None
    gpu_memory_percent: float | None = None
    gpu_temp: float | None = None


class ResourceMonitor:
    """Monitor CPU, memory, and remote GPU resources."""

    def __init__(self, gpu_host: str | None = None, interval: float = 2.0):
        self.gpu_host = gpu_host
        self.interval = interval
        self.snapshots: list[ResourceSnapshot] = []
        self._monitoring = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start monitoring in background."""
        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._task:
            await self._task

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            snapshot = await self._take_snapshot()
            self.snapshots.append(snapshot)
            await asyncio.sleep(self.interval)

    async def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot."""
        # Local CPU/memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        memory_percent = memory.percent

        # Remote GPU (if configured)
        gpu_util = None
        gpu_memory_mb = None
        gpu_memory_percent = None
        gpu_temp = None

        if self.gpu_host:
            try:
                cmd = f"ssh {self.gpu_host} nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    gpu_util = float(parts[0])
                    gpu_memory_mb = float(parts[1])
                    gpu_memory_total = float(parts[2])
                    gpu_memory_percent = (gpu_memory_mb / gpu_memory_total) * 100
                    gpu_temp = float(parts[3])
            except Exception:
                pass  # Silently ignore GPU monitoring failures

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_util=gpu_util,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temp=gpu_temp,
        )

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics."""
        if not self.snapshots:
            return {}

        cpu_avg = sum(s.cpu_percent for s in self.snapshots) / len(self.snapshots)
        cpu_max = max(s.cpu_percent for s in self.snapshots)
        mem_avg = sum(s.memory_mb for s in self.snapshots) / len(self.snapshots)
        mem_max = max(s.memory_mb for s in self.snapshots)

        summary = {
            "cpu_avg": cpu_avg,
            "cpu_max": cpu_max,
            "memory_avg_mb": mem_avg,
            "memory_max_mb": mem_max,
        }

        # Add GPU stats if available
        gpu_snapshots = [s for s in self.snapshots if s.gpu_util is not None]
        if gpu_snapshots:
            summary["gpu_util_avg"] = sum(s.gpu_util for s in gpu_snapshots) / len(gpu_snapshots)
            summary["gpu_util_max"] = max(s.gpu_util for s in gpu_snapshots)
            summary["gpu_memory_avg_mb"] = sum(s.gpu_memory_mb for s in gpu_snapshots) / len(gpu_snapshots)
            summary["gpu_memory_max_mb"] = max(s.gpu_memory_mb for s in gpu_snapshots)
            summary["gpu_temp_avg"] = sum(s.gpu_temp for s in gpu_snapshots) / len(gpu_snapshots)
            summary["gpu_temp_max"] = max(s.gpu_temp for s in gpu_snapshots)

        return summary
