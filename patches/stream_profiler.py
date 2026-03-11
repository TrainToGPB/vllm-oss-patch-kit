# SPDX-License-Identifier: Apache-2.0
"""
Lightweight streaming profiler for diagnosing per-request performance.
Enable via environment variable: VLLM_STREAM_PROFILE=1

Logs a JSON summary per request with:
  - render_ms:        prompt rendering + tokenization time
  - engine_wait_ms:   total time waiting for engine output (GPU decode)
  - py_process_ms:    total Python processing time (harmony parser + delta + SSE)
  - first_token_ms:   time to first engine output (TTFT at engine level)
  - channel_switch_ms: time from first token to first "final" channel token
  - analysis_tokens:  number of tokens in analysis (thinking) channel
  - final_tokens:     number of tokens in final (content) channel
  - other_tokens:     tokens in other channels (commentary, control, etc.)
  - total_tokens:     total output tokens
  - total_ms:         wall clock time for the entire streaming response
  - num_engine_steps: number of engine output batches received
  - json_serial_ms:   total time spent in model_dump_json (SSE serialization)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("vllm.stream_profiler")

ENABLED = os.environ.get("VLLM_STREAM_PROFILE", "0") == "1"


@dataclass
class StreamProfile:
    request_id: str = ""

    # Wall clock
    t_start: float = 0.0
    t_first_engine_output: float = 0.0
    t_first_final_token: float = 0.0
    t_end: float = 0.0

    # Accumulated durations (seconds)
    render_time: float = 0.0
    engine_wait_time: float = 0.0
    py_process_time: float = 0.0
    json_serial_time: float = 0.0

    # Token counters
    analysis_tokens: int = 0
    final_tokens: int = 0
    commentary_tokens: int = 0
    other_tokens: int = 0
    total_tokens: int = 0

    # Step counters
    num_engine_steps: int = 0
    num_yields: int = 0

    # Internal state
    _engine_wait_start: float = 0.0
    _py_start: float = 0.0
    _saw_first_output: bool = False
    _saw_first_final: bool = False

    # Per-step timing for percentile analysis
    engine_wait_steps: list[float] = field(default_factory=list)
    py_process_steps: list[float] = field(default_factory=list)

    def start(self):
        self.t_start = time.perf_counter()

    def mark_render_done(self, render_start: float):
        self.render_time = time.perf_counter() - render_start

    def begin_engine_wait(self):
        self._engine_wait_start = time.perf_counter()

    def end_engine_wait(self):
        now = time.perf_counter()
        step_wait = now - self._engine_wait_start
        self.engine_wait_time += step_wait
        self.engine_wait_steps.append(step_wait)
        self.num_engine_steps += 1
        if not self._saw_first_output:
            self._saw_first_output = True
            self.t_first_engine_output = now

    def begin_py_process(self):
        self._py_start = time.perf_counter()

    def end_py_process(self):
        step_py = time.perf_counter() - self._py_start
        self.py_process_time += step_py
        self.py_process_steps.append(step_py)

    def record_tokens(self, channel: str | None, count: int):
        self.total_tokens += count
        if channel == "analysis":
            self.analysis_tokens += count
            return
        if channel == "final":
            if not self._saw_first_final and count > 0:
                self._saw_first_final = True
                self.t_first_final_token = time.perf_counter()
            self.final_tokens += count
            return
        if channel == "commentary":
            self.commentary_tokens += count
            return
        self.other_tokens += count

    def record_json_serial(self, start: float):
        self.json_serial_time += time.perf_counter() - start

    def finish_and_log(self):
        self.t_end = time.perf_counter()
        total_ms = (self.t_end - self.t_start) * 1000
        first_tok_ms = (
            (self.t_first_engine_output - self.t_start) * 1000
            if self._saw_first_output else -1
        )
        chan_switch_ms = (
            (self.t_first_final_token - self.t_first_engine_output) * 1000
            if self._saw_first_final and self._saw_first_output else -1
        )

        # Percentiles for engine wait
        p50_ew = p99_ew = max_ew = 0.0
        if self.engine_wait_steps:
            s = sorted(self.engine_wait_steps)
            p50_ew = s[len(s) // 2] * 1000
            p99_ew = s[int(len(s) * 0.99)] * 1000
            max_ew = s[-1] * 1000

        # Percentiles for py process
        p50_py = p99_py = max_py = 0.0
        if self.py_process_steps:
            s = sorted(self.py_process_steps)
            p50_py = s[len(s) // 2] * 1000
            p99_py = s[int(len(s) * 0.99)] * 1000
            max_py = s[-1] * 1000

        unaccounted = total_ms - (
            self.render_time * 1000
            + self.engine_wait_time * 1000
            + self.py_process_time * 1000
        )

        summary = {
            "request_id": self.request_id,
            "total_ms": round(total_ms, 2),
            "render_ms": round(self.render_time * 1000, 2),
            "engine_wait_ms": round(self.engine_wait_time * 1000, 2),
            "py_process_ms": round(self.py_process_time * 1000, 2),
            "json_serial_ms": round(self.json_serial_time * 1000, 2),
            "unaccounted_ms": round(unaccounted, 2),
            "first_token_ms": round(first_tok_ms, 2),
            "channel_switch_ms": round(chan_switch_ms, 2),
            "analysis_tokens": self.analysis_tokens,
            "final_tokens": self.final_tokens,
            "commentary_tokens": self.commentary_tokens,
            "other_tokens": self.other_tokens,
            "total_tokens": self.total_tokens,
            "num_engine_steps": self.num_engine_steps,
            "num_yields": self.num_yields,
            "engine_wait_p50_ms": round(p50_ew, 3),
            "engine_wait_p99_ms": round(p99_ew, 3),
            "engine_wait_max_ms": round(max_ew, 3),
            "py_process_p50_ms": round(p50_py, 3),
            "py_process_p99_ms": round(p99_py, 3),
            "py_process_max_ms": round(max_py, 3),
        }

        logger.warning("STREAM_PROFILE %s", summary)


_NOOP = StreamProfile()


def create_profile(request_id: str = "") -> StreamProfile:
    if not ENABLED:
        return _NOOP
    p = StreamProfile(request_id=request_id)
    return p


def is_enabled() -> bool:
    return ENABLED
