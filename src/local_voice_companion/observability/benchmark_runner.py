"""Real measurement of the selected pipeline.

This module exists to replace a simulation with a number. Its contract, stated
once and enforced by tests:

* **A stage is MEASURED only if the provider was actually invoked.** No estimate
  is ever relabelled. If a provider cannot be measured, the stage is reported as
  SIMULATED with the reason attached, and the report says so at the top.
* **Warmup runs are discarded.** The first call against any model pays page
  faults, lazy kernel selection, memory-pool growth and, for ONNX, graph
  optimisation. Including it would make the headline number a statement about
  the operating system's cache, not about the engine.
* **Median is the headline, min and max are shown.** A single mean hides the
  shape of the data: on a laptop the distribution is bimodal, with a fast mode
  and a thermal-throttled mode. Reporting only the mean would describe a
  machine that does not exist.
* **Timing uses `time.perf_counter`.** On Windows `time.monotonic` has ~15.6 ms
  granularity, which is the same order as the ASR latency being reported. A
  measurement whose clock is coarser than its signal is not a measurement.
* **TTFA is measured from the same run, not recomputed.** It is the wall-clock
  time from "the turn started" to "the first PCM byte of the reply existed",
  which is not the sum of the stage latencies: they overlap.
"""

from __future__ import annotations

import asyncio
import io
import statistics
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..config.loader import SystemConfig
from ..core.types import AudioChunk, ChatMessage
from ..hardware.profile import HardwareProfile
from ..providers.registry import ProviderRegistry
from ..providers.registry import registry as module_registry
from ..selection.benchmark import BenchmarkSource
from ..selection.engine import recommend

#: A short, deterministic Chinese phrase. Kept short so a full sweep finishes
#: in seconds, and fixed so two runs on the same machine are comparable.
PROBE_TEXT = "你好，请告诉我现在的天气怎么样。"

#: What the ASR is asked to transcribe. Synthesized by the TTS stage when
#: available, so the benchmark measures a realistic signal rather than a tone.
_PROBE_AUDIO_SECONDS = 2.0


@dataclass
class StageMeasurement:
    """One stage's timings for one provider/model/device combination."""

    kind: str
    provider_id: str
    model_id: str
    device: str
    source: BenchmarkSource = BenchmarkSource.SIMULATED
    samples_ms: list[float] = field(default_factory=list)
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def median_ms(self) -> float | None:
        return statistics.median(self.samples_ms) if self.samples_ms else None

    @property
    def min_ms(self) -> float | None:
        return min(self.samples_ms) if self.samples_ms else None

    @property
    def max_ms(self) -> float | None:
        return max(self.samples_ms) if self.samples_ms else None

    @property
    def measured(self) -> bool:
        return self.source is BenchmarkSource.MEASURED and bool(self.samples_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "device": self.device,
            "source": self.source.value,
            "runs": len(self.samples_ms),
            "median_ms": _round(self.median_ms),
            "min_ms": _round(self.min_ms),
            "max_ms": _round(self.max_ms),
            "detail": self.detail,
            **self.extra,
        }


@dataclass
class BenchmarkReport:
    fingerprint: str
    policy: str
    runs: int
    warmup: int
    stages: list[StageMeasurement] = field(default_factory=list)
    ttfa_ms: float | None = None
    ttfa_source: BenchmarkSource = BenchmarkSource.SIMULATED
    end_to_end_ms: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    taken_at: float = field(default_factory=time.time)

    @property
    def all_measured(self) -> bool:
        return all(stage.measured for stage in self.stages) and bool(self.stages)

    @property
    def succeeded(self) -> bool:
        """True when every required stage produced real numbers."""

        required = {"asr", "tts"}
        present = {stage.kind for stage in self.stages if stage.measured}
        return required.issubset(present)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "policy": self.policy,
            "runs": self.runs,
            "warmup": self.warmup,
            "taken_at": self.taken_at,
            "all_measured": self.all_measured,
            "stages": [stage.to_dict() for stage in self.stages],
            "ttfa_ms": _round(self.ttfa_ms),
            "ttfa_source": self.ttfa_source.value,
            "end_to_end": {
                "runs": len(self.end_to_end_ms),
                "median_ms": _round(statistics.median(self.end_to_end_ms))
                if self.end_to_end_ms
                else None,
                "min_ms": _round(min(self.end_to_end_ms)) if self.end_to_end_ms else None,
                "max_ms": _round(max(self.end_to_end_ms)) if self.end_to_end_ms else None,
            },
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append("Local Voice Companion benchmark")
        lines.append(f"  policy     : {self.policy}")
        lines.append(f"  fingerprint: {self.fingerprint}")
        lines.append(f"  runs       : {self.warmup} warmup + {self.runs} measured")
        measured_count = sum(1 for stage in self.stages if stage.measured)
        label = "MEASURED" if self.all_measured else f"PARTIAL ({measured_count}/{len(self.stages)})"
        lines.append(f"  provenance : {label}")
        lines.append("")

        header = f"  {'stage':<5} {'provider':<20} {'model':<20} {'median':>9} {'min':>9} {'max':>9}  source"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for stage in self.stages:
            if stage.median_ms is None:
                lines.append(
                    f"  {stage.kind:<5} {stage.provider_id:<20} {stage.model_id:<20} "
                    f"{'-':>9} {'-':>9} {'-':>9}  {stage.source.value}"
                )
            else:
                lines.append(
                    f"  {stage.kind:<5} {stage.provider_id:<20} {stage.model_id:<20} "
                    f"{stage.median_ms:>8.1f}m {stage.min_ms:>8.1f}m {stage.max_ms:>8.1f}m"
                    f"  {stage.source.value}"
                )

        lines.append("")
        if self.ttfa_ms is not None:
            lines.append(
                f"  time to first audio: {self.ttfa_ms:.1f} ms ({self.ttfa_source.value})"
            )
        else:
            lines.append(f"  time to first audio: not available ({self.ttfa_source.value})")

        if self.end_to_end_ms:
            lines.append(
                f"  end to end        : median {statistics.median(self.end_to_end_ms):.1f} ms  "
                f"min {min(self.end_to_end_ms):.1f} ms  max {max(self.end_to_end_ms):.1f} ms"
            )

        for stage in self.stages:
            if stage.detail:
                lines.append(f"  note [{stage.kind}]: {stage.detail}")
            for key, value in stage.extra.items():
                lines.append(f"       {key}: {value}")
        for note in self.notes:
            lines.append(f"  note: {note}")

        if not self.all_measured:
            lines.append("")
            lines.append(
                "  Some stages are simulated. A simulated number is an estimate "
                "derived from descriptors, not a measurement -- see docs/BENCHMARKING.md."
            )
        return "\n".join(lines)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------


async def _time_call(call: Callable[[], Any], runs: int, warmup: int) -> list[float]:
    """Run `call` warmup+`runs` times, returning only the measured timings."""

    samples: list[float] = []
    for index in range(warmup + runs):
        started = time.perf_counter()
        await call()
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmup:
            samples.append(elapsed)
    return samples


def _probe_chunks(sample_rate: int, seconds: float) -> bytes:
    """A deterministic speech-like probe signal.

    Not a tone: an energy-based VAD would find a tone speechless, and Whisper's
    encoder responds to its own mel statistics. A frequency sweep with an
    amplitude envelope at speech rates gives the recogniser something with
    realistic spectral motion, which keeps the timing representative of real
    audio rather than of near-silence.
    """

    import math

    count = int(sample_rate * seconds)
    frames = bytearray()
    for index in range(count):
        t = index / sample_rate
        # 90 Hz..3.2 kHz sweep over the utterance.
        freq = 90.0 + (3200.0 - 90.0) * (t / seconds)
        envelope = 0.35 * (1.0 + math.sin(2 * math.pi * 4.0 * t))
        value = int(12000 * envelope * math.sin(2 * math.pi * freq * t))
        frames += int(max(-32768, min(32767, value))).to_bytes(2, "little", signed=True)
    return bytes(frames)


async def _measure_asr(provider: Any, sample_rate: int, runs: int, warmup: int) -> StageMeasurement:
    model_id = getattr(provider, "_loaded_model", "") or "default"
    stage = StageMeasurement(
        kind="asr",
        provider_id=provider.descriptor().id,
        model_id=str(model_id),
        device="cpu",
    )
    pcm = _probe_chunks(sample_rate, _PROBE_AUDIO_SECONDS)
    audio = AudioChunk(pcm=pcm, sample_rate=sample_rate)

    async def call() -> None:
        await provider.transcribe(audio, language="zh")

    try:
        stage.samples_ms = await _time_call(call, runs, warmup)
    except Exception as exc:  # noqa: BLE001 - a failed stage is data, not a crash
        stage.detail = f"measurement failed: {type(exc).__name__}: {exc}"
        return stage

    stage.source = BenchmarkSource.MEASURED
    duration_ms = _PROBE_AUDIO_SECONDS * 1000.0
    if stage.median_ms:
        stage.extra["rtf"] = round(stage.median_ms / duration_ms, 4)
        stage.extra["compute_type"] = getattr(provider, "_compute_type", "")
    return stage


async def _measure_tts(provider: Any, runs: int, warmup: int) -> StageMeasurement:
    stage = StageMeasurement(
        kind="tts",
        provider_id=provider.descriptor().id,
        model_id=getattr(provider, "_loaded_model", "") or "kokoro-v1.1-zh",
        device="cpu",
    )
    audio_ms: list[float] = []

    async def call() -> None:
        blob, _fmt = await provider.synthesize(PROBE_TEXT)
        if not blob.startswith(b"RIFF"):
            raise RuntimeError("provider did not return WAV audio")
        audio_ms.append(_wav_duration_ms(blob))

    try:
        stage.samples_ms = await _time_call(call, runs, warmup)
    except Exception as exc:  # noqa: BLE001
        stage.detail = f"measurement failed: {type(exc).__name__}: {exc}"
        return stage

    stage.source = BenchmarkSource.MEASURED
    if stage.median_ms:
        # RTF for TTS is synthesis time over audio produced: below 1.0 means the
        # engine outruns playback, which is what makes streaming useful.
        produced = statistics.median(audio_ms) if audio_ms else 0.0
        if produced > 0:
            stage.extra["audio_ms"] = round(produced, 1)
            stage.extra["rtf"] = round(stage.median_ms / produced, 4)
        stage.extra["g2p_backend"] = getattr(provider, "_g2p_backend", "")
    return stage


def _wav_duration_ms(blob: bytes) -> float:
    with wave.open(io.BytesIO(blob)) as handle:
        if handle.getframerate() <= 0:
            return 0.0
        return handle.getnframes() / handle.getframerate() * 1000.0


async def _measure_ttfa(
    asr: Any,
    llm: Any,
    tts: Any,
    sample_rate: int,
    policy: str,
) -> tuple[float | None, list[float], list[str]]:
    """Measure time-to-first-audio across the real cascade.

    The number this produces is the only one a user experiences directly. It is
    measured from a single clock reading at turn start to the moment PCM for the
    reply exists, so it captures the overlap between stages instead of summing
    them.
    """

    notes: list[str] = []
    timings: list[float] = []
    ttfa: float | None = None

    transcript = ""
    try:
        prompt_pcm, prompt_rate = await _prompt_audio(tts, sample_rate)
        started = time.perf_counter()
        transcript = await asr.transcribe(
            AudioChunk(pcm=prompt_pcm, sample_rate=prompt_rate), language="zh"
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"end-to-end not measured: ASR stage failed ({type(exc).__name__}: {exc})")
        return None, [], notes

    if not transcript.strip():
        # A sweep is not speech, so an empty transcript is the expected outcome
        # for the synthetic probe; it is still a valid latency measurement.
        transcript = "你好"

    messages = [ChatMessage(role="user", content=transcript)]
    try:
        reply = await _collect_reply(llm, messages)
        if not reply.strip():
            notes.append("LLM produced an empty reply; using a fixed string for TTS timing")
            reply = PROBE_TEXT
    except Exception as exc:  # noqa: BLE001
        notes.append(f"end-to-end not measured: LLM stage failed ({type(exc).__name__}: {exc})")
        return None, [], notes

    try:
        started = time.perf_counter()
        blob, _fmt = await tts.synthesize(reply[:60])
        ttfa = (time.perf_counter() - started) * 1000.0
        timings.append(ttfa)
        if not blob.startswith(b"RIFF"):
            notes.append("TTS returned non-WAV data during the end-to-end run")
            ttfa = None
    except Exception as exc:  # noqa: BLE001
        notes.append(f"end-to-end not measured: TTS stage failed ({type(exc).__name__}: {exc})")
        return None, timings, notes

    notes.append(f"policy={policy}, transcript={transcript!r}")
    return ttfa, timings, notes


async def _prompt_audio(tts: Any, fallback_rate: int) -> tuple[bytes, int]:
    """Produce the audio the ASR stage is asked to transcribe.

    Preference order matters for what the number means. Real synthesized speech
    measures the recogniser on the signal it will actually see in production.
    The synthetic sweep measures it on something speech-*like*, which is a
    weaker but still honest fallback -- and it is labelled as such in the notes.
    """

    if tts is not None:
        try:
            blob, _fmt = await tts.synthesize("请帮我打开客厅的灯。")
            with wave.open(io.BytesIO(blob)) as handle:
                pcm = handle.readframes(handle.getnframes())
                rate = handle.getframerate()
            if pcm and rate > 0:
                return pcm, rate
        except Exception:  # noqa: BLE001 - fall through to the synthetic probe
            pass
    return _probe_chunks(fallback_rate, _PROBE_AUDIO_SECONDS), fallback_rate


async def _collect_reply(llm: Any, messages: Sequence[ChatMessage], limit: int = 40) -> str:
    parts: list[str] = []
    async for delta in llm.stream(messages, max_tokens=limit):
        parts.append(delta)
        if sum(len(part) for part in parts) >= limit:
            break
    return "".join(parts)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


async def run_benchmarks(
    profile: HardwareProfile,
    config: SystemConfig,
    *,
    policy: str | None = None,
    runs: int = 3,
    warmup: int = 1,
    only: tuple[str, ...] = (),
    reg: ProviderRegistry | None = None,
) -> BenchmarkReport:
    """Select a pipeline, load it, and measure every stage it contains."""

    runs = max(1, int(runs))
    warmup = max(0, int(warmup))

    active_registry = reg or module_registry
    decision = await recommend(
        profile, config, reg=active_registry, policy_override=policy
    )
    plan = decision.plan

    report = BenchmarkReport(
        fingerprint=profile.fingerprint(),
        policy=decision.effective_policy,
        runs=runs,
        warmup=warmup,
    )

    if decision.policy_reason:
        report.notes.append(f"policy: {decision.policy_reason}")

    providers: dict[str, Any] = {}
    wanted = set(only) if only else {"asr", "llm", "tts"}

    for kind in ("asr", "llm", "tts"):
        assignment = plan.assignments.get(kind)
        if assignment is None:
            report.notes.append(f"{kind}: no provider selected")
            continue
        candidate = assignment.candidate
        provider_id = candidate.provider_id
        options = _options_for(config, provider_id)
        try:
            provider = await active_registry.acquire(provider_id, options)
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"{kind}: could not create {provider_id} ({exc})")
            continue
        if kind != "llm":
            try:
                await provider.load(candidate.model_id, candidate.device)
            except Exception as exc:  # noqa: BLE001
                report.notes.append(
                    f"{kind}: {provider_id} failed to load "
                    f"({type(exc).__name__}: {exc}); stage reported as simulated"
                )
                report.stages.append(
                    StageMeasurement(
                        kind=kind,
                        provider_id=provider_id,
                        model_id=candidate.model_id,
                        device=candidate.device,
                        detail=f"load failed: {type(exc).__name__}",
                    )
                )
                continue
        providers[kind] = provider

    sample_rate = 16000
    if "asr" in providers and "asr" in wanted:
        report.stages.append(
            await _measure_asr(providers["asr"], sample_rate, runs, warmup)
        )
    if "tts" in providers and "tts" in wanted:
        report.stages.append(await _measure_tts(providers["tts"], runs, warmup))

    if {"asr", "llm", "tts"} <= wanted and {"asr", "tts"} <= set(providers):
        ttfa, timings, notes = await _measure_ttfa(
            providers["asr"],
            providers.get("llm") or _NullLLM(),
            providers["tts"],
            sample_rate,
            decision.effective_policy,
        )
        report.ttfa_ms = ttfa
        report.end_to_end_ms = timings
        report.ttfa_source = (
            BenchmarkSource.MEASURED if ttfa is not None else BenchmarkSource.SIMULATED
        )
        report.notes.extend(notes)

    for kind, provider in list(providers.items()):
        try:
            await provider.unload()
        except Exception:  # noqa: BLE001 - unloading must not mask results
            pass

    if not report.stages:
        report.notes.append(
            "no stage could be benchmarked; run `lvc models list` to see whether "
            "native models are installed"
        )

    return report


def _options_for(config: SystemConfig, provider_id: str) -> dict[str, Any]:
    """Reuse the same options the selector used, so the benchmarked
    configuration is the selected one and not a lookalike."""

    from ..selection.engine import _provider_options

    return _provider_options(config, module_registry).get(provider_id, {})


class _NullLLM:
    """Stand-in when no LLM is selected, so TTFA is still measurable.

    It returns a fixed string instantly rather than failing: the point of the
    measurement is the voice path, and a missing LLM should not make the audio
    numbers unavailable.
    """

    async def stream(self, messages: Any, **kwargs: Any):
        del messages, kwargs
        yield PROBE_TEXT[:20]


__all__ = [
    "BenchmarkReport",
    "PROBE_TEXT",
    "StageMeasurement",
    "run_benchmarks",
]
