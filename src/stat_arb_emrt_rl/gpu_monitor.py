# ───────────────────────────────────────────────
# GPU Monitoring Utility
# ───────────────────────────────────────────────
import subprocess
import time
from typing import Dict, Optional


def get_gpu_stats() -> Optional[Dict]:
    """Get current GPU temperature, utilization, and memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(", ")
        if len(parts) < 4:
            return None

        return {
            "temp_c": int(parts[0]),
            "util_pct": int(parts[1]),
            "mem_used_mb": int(parts[2]),
            "mem_total_mb": int(parts[3]),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_gpu_safe(max_temp: int = 80, max_util: int = 95) -> bool:
    """Check if GPU is within safe operating limits."""
    stats = get_gpu_stats()
    if stats is None:
        return True  # No GPU = no concern

    if stats["temp_c"] >= max_temp:
        print(f"GPU TEMP WARNING: {stats['temp_c']}C >= {max_temp}C limit")
        return False

    if stats["util_pct"] >= max_util:
        print(f"GPU UTIL WARNING: {stats['util_pct']}% >= {max_util}% limit")
        return False

    return True


def print_gpu_status():
    """Print formatted GPU status line."""
    stats = get_gpu_stats()
    if stats is None:
        print("GPU: not available")
        return

    temp = stats["temp_c"]
    temp_color = "\033[92m" if temp < 60 else "\033[93m" if temp < 75 else "\033[91m"
    print(f"GPU: {temp_color}{temp}C\033[0m | "
          f"{stats['util_pct']}% util | "
          f"{stats['mem_used_mb']}/{stats['mem_total_mb']} MB")
