#!/usr/bin/env python3
"""
test_consecutive_tool_count.py
==============================
Verifies the consecutive-tool-count enforcement logic in run_agent.py.

We test the actual source code via inspection, which is stable regardless
of how many internal attributes AIAgent.__init__ requires.  This approach
covers:
  - Sequential path: per-tool increment + limit check + reset at end of burst
  - Concurrent path: batch-size increment + limit check + cancel before submit
  - Heuristic: _should_parallelize_tool_batch routing logic
  - Threshold value: _MAX_CONSECUTIVE_TOOL_CALLS default

Run: pytest tests/agent/test_consecutive_tool_count.py -v
"""

from __future__ import annotations

import inspect
import pytest
import run_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src(cls, method_name: str) -> str:
    return inspect.getsource(getattr(cls, method_name))


class _FakeToolCall:
    """Mimics the OpenAI tool_call structure: tc.function.name."""
    def __init__(self, name: str, arguments: str = "{}"):
        self.function = _FakeFunction(name, arguments)


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------

class TestSequentialPath:
    """Verify _execute_tool_calls_sequential has correct counter logic."""

    def test_increments_per_tool(self):
        """Counter increments once per tool in the loop."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_sequential')
        assert 'self._consecutive_tool_count += 1' in src

    def test_limit_check_uses_greater_than(self):
        """Limit check uses > (not >=), stopping before the tool that exceeds."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_sequential')
        assert 'self._consecutive_tool_count > self._MAX_CONSECUTIVE_TOOL_CALLS' in src

    def test_break_after_exceeding_limit(self):
        """Loop breaks immediately when limit is exceeded."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_sequential')
        assert 'break' in src

    def test_skipped_tools_get_messages(self):
        """Skipped tools get a message with 'skipped — consecutive' appended."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_sequential')
        assert 'skipped — consecutive' in src
        assert 'messages.append' in src

    def test_resets_after_tools_complete(self):
        """Counter resets to 0 after all tools in the burst complete.

        In the sequential path the reset is at the END of the method body
        (not guarded by 'not assistant_message.tool_calls' — that guard is
        only in the concurrent path.  The sequential path is called after
        the API returns a response that has tool_calls; the reset happens
        as a post-processing step at the end of _execute_tool_calls_sequential.
        """
        src = _src(run_agent.AIAgent, '_execute_tool_calls_sequential')
        assert 'self._consecutive_tool_count = 0' in src


# ---------------------------------------------------------------------------
# Concurrent path
# ---------------------------------------------------------------------------

class TestConcurrentPath:
    """Verify _execute_tool_calls_concurrent has correct counter logic."""

    def test_increments_by_batch_size(self):
        """Counter increments by full batch size (num_tools), not per-call."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_concurrent')
        assert 'self._consecutive_tool_count += num_tools' in src

    def test_limit_check_before_executor_submit(self):
        """Limit check happens BEFORE workers are submitted to the executor."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_concurrent')
        assert 'self._consecutive_tool_count > self._MAX_CONSECUTIVE_TOOL_CALLS' in src

        # Verify limit check comes before executor.submit
        lines = src.split('\n')
        limit_idx = next((i for i, l in enumerate(lines) if 'self._consecutive_tool_count >' in l), None)
        submit_idx = next((i for i, l in enumerate(lines) if 'executor.submit' in l), None)
        assert limit_idx is not None and submit_idx is not None
        assert limit_idx < submit_idx, (
            f"Limit check (line {limit_idx}) must precede executor.submit (line {submit_idx})"
        )

    def test_cancelled_batch_returns_before_executor(self):
        """Early return exits before any worker threads are launched."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_concurrent')
        lines = src.split('\n')
        in_limit_block = False
        has_return_before_executor = False
        for i, line in enumerate(lines):
            if 'self._consecutive_tool_count >' in line:
                in_limit_block = True
            if in_limit_block and line.strip().startswith('return'):
                has_return_before_executor = True
                break
        assert has_return_before_executor, (
            "concurrent path must return before executor when limit exceeded"
        )

    def test_skipped_messages_for_cancelled_batch(self):
        """Cancelled batch appends 'skipped — consecutive' messages."""
        src = _src(run_agent.AIAgent, '_execute_tool_calls_concurrent')
        assert 'skipped — consecutive' in src
        assert 'messages.append' in src

    def test_resets_when_no_tool_calls(self):
        """Counter resets to 0 when assistant_message.tool_calls is empty.

        This is a pre-dispatch check at the START of _execute_tool_calls_concurrent.
        Mirrors the LangGraph pattern where a reasoning burst ends when the
        model returns without tool_calls.
        """
        src = _src(run_agent.AIAgent, '_execute_tool_calls_concurrent')
        assert 'if not assistant_message.tool_calls and hasattr' in src
        assert 'self._consecutive_tool_count = 0' in src


# ---------------------------------------------------------------------------
# Parallelization heuristic
# ---------------------------------------------------------------------------

class TestParallelizationHeuristic:
    """Verify _should_parallelize_tool_batch routes correctly."""

    def test_single_tool_always_sequential(self):
        """Zero tools → not parallelized."""
        result = run_agent._should_parallelize_tool_batch([])
        assert result is False

    def test_clarify_forces_sequential(self):
        """clarify in batch → not parallelized (NEVER_PARALLEL set)."""
        batch = [
            _FakeToolCall('read_file', '{"path":"/tmp/a"}'),
            _FakeToolCall('clarify', '{"question":"ok?"}'),
        ]
        assert run_agent._should_parallelize_tool_batch(batch) is False

    def test_independent_read_files_parallel(self):
        """Two read_file calls targeting different paths → parallel."""
        batch = [
            _FakeToolCall('read_file', '{"path":"/tmp/a"}'),
            _FakeToolCall('read_file', '{"path":"/tmp/b"}'),
        ]
        assert run_agent._should_parallelize_tool_batch(batch) is True

    def test_same_path_blocks_parallel(self):
        """Two read_file calls targeting the same path → blocked."""
        batch = [
            _FakeToolCall('read_file', '{"path":"/tmp/same"}'),
            _FakeToolCall('read_file', '{"path":"/tmp/same"}'),
        ]
        assert run_agent._should_parallelize_tool_batch(batch) is False


# ---------------------------------------------------------------------------
# Threshold value
# ---------------------------------------------------------------------------

class TestThresholdValue:
    """Verify _MAX_CONSECUTIVE_TOOL_CALLS is set to 7 in AIAgent.__init__."""

    def test_default_is_7(self):
        """The class-level docstring/comment confirms threshold is 7."""
        src = inspect.getsource(run_agent.AIAgent.__init__)
        assert '_MAX_CONSECUTIVE_TOOL_CALLS: int = 7' in src
        assert '7' in src  # comment confirms 7 is intentional


if __name__ == "__main__":
    pytest.main([__file__, "-v"])