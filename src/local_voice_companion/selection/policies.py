"""Policy weight lookup and AUTO -> concrete policy resolution.

All tunable scoring numbers live here (or in config/defaults.py). The selector
must not contain magic constants.
"""

from __future__ import annotations

from typing import Any

from ..config.defaults import AUTO_THRESHOLDS, policy_weights
from ..core.types import SelectionPolicy
from ..hardware.profile import HardwareProfile


def resolve_policy(
    requested: SelectionPolicy | str,
    profile: HardwareProfile,
    *,
    cpu_only: bool = False,
    prefer_local: bool = True,
) -> tuple[SelectionPolicy, str]:
    """Return (effective policy, human explanation).

    AUTO deliberately collapses into one concrete policy so the UI can state
    exactly *why* a configuration was chosen.
    """

    policy = SelectionPolicy(requested)
    if policy is not SelectionPolicy.AUTO:
        return policy, f"explicitly requested {policy.value}"

    gpu = profile.primary_gpu
    vram = gpu.vram_total_mb if gpu else 0
    threads = profile.cpu.threads
    ram = profile.memory.total_mb

    if cpu_only or not profile.has_accelerator or vram == 0:
        return (
            SelectionPolicy.CPU_ONLY,
            "no usable accelerator detected; falling back to CPU providers",
        )
    if vram <= AUTO_THRESHOLDS["tight_vram_mb"]:
        return (
            SelectionPolicy.LOW_MEMORY,
            f"{vram} MB VRAM detected ({AUTO_THRESHOLDS['tight_vram_mb']} MB threshold "
            "for comfortable local inference); prioritising memory safety",
        )
    if threads and threads <= AUTO_THRESHOLDS["weak_cpu_threads"]:
        return (
            SelectionPolicy.LOW_MEMORY,
            f"only {threads} CPU threads available; keeping CPU-side work light",
        )
    if vram >= AUTO_THRESHOLDS["high_end_vram_mb"] and (not ram or ram >= 16384):
        return (
            SelectionPolicy.QUALITY,
            f"{vram} MB VRAM and ample RAM detected; prioritising output quality",
        )
    return SelectionPolicy.BALANCED, "balanced default for a mid-range local machine"


def weights_for(policy: SelectionPolicy | str) -> dict[str, float]:
    return policy_weights(policy)


def describe_policy(policy: SelectionPolicy | str) -> dict[str, Any]:
    policy = SelectionPolicy(policy)
    return {
        "id": policy.value,
        "weights": weights_for(policy),
        "summary": {
            SelectionPolicy.ULTRA_LOW_LATENCY.value: "最短首音延迟，牺牲一部分质量",
            SelectionPolicy.BALANCED.value: "默认：延迟、质量与资源安全均衡",
            SelectionPolicy.QUALITY.value: "最佳语音与语言质量，延迟更高",
            SelectionPolicy.LOW_MEMORY.value: "显存/内存紧张时使用",
            SelectionPolicy.CPU_ONLY.value: "无可用加速器",
            SelectionPolicy.AUTO.value: "根据本机硬件自动选择具体策略",
            SelectionPolicy.MANUAL.value: "完全手动指定的 provider/model/device",
        }.get(policy.value, ""),
    }
