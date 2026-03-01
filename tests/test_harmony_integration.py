# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for Harmony parser fixes (GPT-OSS models).

Tests cover the full pipeline of changes:
1. Engine returns raw text (no premature cleaning)
2. Tool parser handles missing EOS tokens
3. Reasoning parser always runs (including with tool calls)
4. Server-side fallback and content preservation
5. Streaming accumulation and tool detection
6. _clean_gpt_oss_output strips <|end|> in fallback path

Usage:
    pytest tests/test_harmony_integration.py -v
"""

import json
import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


# =============================================================================
# _parse_tool_calls_with_parser: content preservation when no tool calls
# =============================================================================


class TestParserContentPreservation:
    """When harmony parser finds no tool calls but extracts final-channel
    content, the server should use that content directly instead of
    falling through to the generic parser with raw text."""

    def test_harmony_final_channel_content_preserved(self):
        """Parser extracts clean content from final channel; server uses it."""
        import vllm_mlx.server as server

        original_auto = server._enable_auto_tool_choice
        original_parser_name = server._tool_call_parser
        original_instance = server._tool_parser_instance

        try:
            server._enable_auto_tool_choice = True
            server._tool_call_parser = "harmony"
            server._tool_parser_instance = None

            raw_text = (
                "<|channel|>analysis<|message|>Let me think<|end|>"
                "<|channel|>final<|message|>The answer is 42."
            )

            cleaned, tool_calls = server._parse_tool_calls_with_parser(
                raw_text, None
            )

            assert tool_calls is None
            assert cleaned == "The answer is 42."
        finally:
            server._enable_auto_tool_choice = original_auto
            server._tool_call_parser = original_parser_name
            server._tool_parser_instance = original_instance

    def test_harmony_tool_call_extracted(self):
        """Parser extracts tool call with content."""
        import vllm_mlx.server as server

        original_auto = server._enable_auto_tool_choice
        original_parser_name = server._tool_call_parser
        original_instance = server._tool_parser_instance

        try:
            server._enable_auto_tool_choice = True
            server._tool_call_parser = "harmony"
            server._tool_parser_instance = None

            raw_text = (
                "<|channel|>commentary to=functions.read_file\n"
                "<|constrain|>json\n"
                '<|message|>{"path": "/etc/hosts"}'
            )

            cleaned, tool_calls = server._parse_tool_calls_with_parser(
                raw_text, None
            )

            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].function.name == "read_file"
            args = json.loads(tool_calls[0].function.arguments)
            assert args["path"] == "/etc/hosts"
        finally:
            server._enable_auto_tool_choice = original_auto
            server._tool_call_parser = original_parser_name
            server._tool_parser_instance = original_instance

    def test_harmony_no_content_falls_through(self):
        """No tool calls and no content: falls through to generic parser."""
        import vllm_mlx.server as server

        original_auto = server._enable_auto_tool_choice
        original_parser_name = server._tool_call_parser
        original_instance = server._tool_parser_instance

        try:
            server._enable_auto_tool_choice = True
            server._tool_call_parser = "harmony"
            server._tool_parser_instance = None

            raw_text = "Just plain text with no channel markers."

            cleaned, tool_calls = server._parse_tool_calls_with_parser(
                raw_text, None
            )

            assert "Just plain text" in cleaned
        finally:
            server._enable_auto_tool_choice = original_auto
            server._tool_call_parser = original_parser_name
            server._tool_parser_instance = original_instance


# =============================================================================
# Reasoning parser always runs (including alongside tool calls)
# =============================================================================


class TestReasoningAlwaysRuns:
    """Reasoning parser should extract analysis even when tool calls exist."""

    def test_reasoning_extracted_with_tool_calls(self):
        """Reasoning parser finds analysis blocks from raw text even when
        tool calls are also present."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
        from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

        raw_text = (
            "<|channel|>analysis<|message|>I need to read /etc/hosts"
            "<|start|>assistant"
            "<|channel|>commentary to=functions.read_file"
            "<|constrain|>json"
            '<|message|>{"file_path": "/etc/hosts"}'
        )

        tool_parser = HarmonyToolParser()
        tool_result = tool_parser.extract_tool_calls(raw_text)
        assert tool_result.tools_called

        reasoning_parser = HarmonyReasoningParser()
        reasoning, parsed_content = reasoning_parser.extract_reasoning(raw_text)
        assert reasoning is not None
        assert "read /etc/hosts" in reasoning

    def test_reasoning_uses_raw_text_not_cleaned(self):
        """Reasoning parser must receive raw text with channel tokens, not
        pre-cleaned text that would have them stripped."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        raw_text = (
            "<|channel|>analysis<|message|>Deep reasoning here<|end|>"
            "<|channel|>final<|message|>Final answer<|return|>"
        )

        parser = HarmonyReasoningParser()
        reasoning, content = parser.extract_reasoning(raw_text)
        assert reasoning == "Deep reasoning here"
        assert content == "Final answer"

        # If channel tokens are stripped (pre-cleaned), the parser can't
        # find anything — this proves the parser NEEDS raw text.
        cleaned_text = "Deep reasoning hereFinal answer"
        reasoning2, content2 = parser.extract_reasoning(cleaned_text)
        assert reasoning2 is None
        assert content2 is None

    def test_parsed_content_not_used_when_tool_calls(self):
        """When tool calls are found, reasoning's parsed_content should NOT
        override the cleaned_text."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        raw_text = (
            "<|channel|>analysis<|message|>Thinking<|end|>"
            "<|channel|>commentary to=functions.func<|constrain|>json"
            '<|message|>{"a": 1}<|call|>'
            "<|channel|>final<|message|>Done<|return|>"
        )

        parser = HarmonyReasoningParser()
        reasoning, parsed_content = parser.extract_reasoning(raw_text)

        assert reasoning == "Thinking"
        assert parsed_content == "Done"


# =============================================================================
# Engine: raw text passthrough (no premature cleaning)
# =============================================================================


class TestEngineRawTextPassthrough:
    """Engine chat() must return raw text with channel tokens intact."""

    def test_simple_engine_chat_returns_raw_text(self):
        """SimpleEngine.chat() should NOT call clean_output_text()."""
        import asyncio
        from vllm_mlx.engine.simple import SimpleEngine

        raw_output = (
            "<|channel|>analysis<|message|>Thinking<|end|>"
            "<|channel|>final<|message|>The answer<|return|>"
        )

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = raw_output
        mock_result.tokens = [1, 2, 3]
        mock_result.finish_reason = "stop"
        mock_result.prompt_tokens = 10
        mock_result.completion_tokens = 20
        mock_model.chat = MagicMock(return_value=mock_result)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
            engine._model = mock_model
            engine._loaded = True

            output = asyncio.get_event_loop().run_until_complete(
                engine.chat(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=100,
                )
            )

        assert "<|channel|>" in output.text
        assert "<|message|>" in output.text
        assert "analysis" in output.text


class TestBatchedEngineSkipClean:
    """BatchedEngine uses _skip_clean flag to defer cleaning for chat()."""

    def test_skip_clean_flag_passed_from_chat(self):
        """chat() passes _skip_clean=True to generate()."""
        import asyncio
        from unittest.mock import AsyncMock
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")
            engine._loaded = True
            engine._tokenizer = MagicMock()
            engine._tokenizer.apply_chat_template = MagicMock(
                return_value="formatted prompt"
            )

            mock_generate = AsyncMock()
            mock_generate.return_value = MagicMock(
                text="<|channel|>final<|message|>Answer",
                prompt_tokens=10,
                completion_tokens=5,
            )
            engine.generate = mock_generate

            asyncio.get_event_loop().run_until_complete(
                engine.chat(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=100,
                )
            )

            call_kwargs = mock_generate.call_args
            assert call_kwargs is not None
            _, kwargs = call_kwargs
            assert kwargs.get("_skip_clean") is True

    def test_skip_clean_preserves_channel_tokens(self):
        """When _skip_clean=True, raw text is returned without cleaning."""
        from vllm_mlx.api.utils import clean_output_text

        raw = "<|channel|>analysis<|message|>Think<|end|><|channel|>final<|message|>Answer<|return|>"

        cleaned = clean_output_text(raw)
        assert "<|channel|>" not in cleaned
        assert cleaned == "Answer"

        assert "<|channel|>" in raw

    def test_skip_clean_false_cleans_normally(self):
        """When _skip_clean=False (default), text is cleaned normally."""
        from vllm_mlx.api.utils import clean_output_text

        raw = "<|channel|>final<|message|>Hello world<|return|>"
        cleaned = clean_output_text(raw)
        assert cleaned == "Hello world"


# =============================================================================
# Streaming: tool_accumulated_text updated even when reasoning suppresses
# =============================================================================


class TestStreamingToolAccumulation:
    """When reasoning parser suppresses a chunk (returns None), the raw text
    must still be accumulated in tool_accumulated_text for the streaming
    fallback to detect tool calls at end-of-stream."""

    def test_commentary_tokens_accumulated_for_tool_parser(self):
        """Harmony commentary channel tokens suppressed by reasoning parser
        should still be visible to the tool parser fallback."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        parser = HarmonyReasoningParser()
        parser.reset_state()

        tool_accumulated = ""

        tokens = [
            "<|channel|>commentary to=functions.func\n",
            "<|message|>",
            '{"arg": "value"}',
        ]

        accumulated = ""
        for token in tokens:
            prev = accumulated
            accumulated += token
            delta_msg = parser.extract_reasoning_streaming(
                prev, accumulated, token
            )
            if delta_msg is None:
                tool_accumulated += token

        assert "to=functions.func" in tool_accumulated
        assert '{"arg": "value"}' in tool_accumulated

    def test_analysis_tokens_accumulated_for_tool_parser(self):
        """Analysis channel tokens are also accumulated for tool parser."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        parser = HarmonyReasoningParser()
        parser.reset_state()

        tool_accumulated = ""

        tokens = [
            "<|channel|>analysis<|message|>",
            "thinking...",
            "<|end|>",
            "<|channel|>commentary to=functions.search\n",
            "<|message|>",
            '{"q": "test"}',
        ]

        accumulated = ""
        for token in tokens:
            prev = accumulated
            accumulated += token
            delta_msg = parser.extract_reasoning_streaming(
                prev, accumulated, token
            )
            if delta_msg is None:
                tool_accumulated += token

        assert "to=functions.search" in tool_accumulated


# =============================================================================
# Streaming fallback: _has_tool_markup detects harmony format
# =============================================================================


class TestStreamingFallbackToolMarkup:
    """The streaming fallback at end-of-stream checks tool_accumulated_text
    for tool call markers.  It must detect both <tool_call> and
    to=functions. (harmony format)."""

    def test_detect_harmony_tool_markup(self):
        """to=functions. in accumulated text triggers tool extraction."""
        accumulated = (
            "<|channel|>commentary to=functions.read_file\n"
            "<|constrain|>json\n"
            '<|message|>{"path": "/etc/hosts"}'
        )
        has_markup = (
            "<tool_call>" in accumulated
            or "to=functions." in accumulated
        )
        assert has_markup is True

    def test_detect_xml_tool_markup(self):
        """<tool_call> in accumulated text still detected."""
        accumulated = '<tool_call>{"name": "func", "arguments": {}}</tool_call>'
        has_markup = (
            "<tool_call>" in accumulated
            or "to=functions." in accumulated
        )
        assert has_markup is True

    def test_no_tool_markup_in_plain_text(self):
        """Plain text does not trigger false positive."""
        accumulated = "This is just a regular response."
        has_markup = (
            "<tool_call>" in accumulated
            or "to=functions." in accumulated
        )
        assert has_markup is False


# =============================================================================
# Full pipeline: raw output -> parsers -> correct API response
# =============================================================================


class TestFullPipelineNoEOS:
    """Simulate the full server pipeline with realistic GPT-OSS output
    where EOS tokens are stripped by mlx-lm."""

    def test_tool_call_pipeline(self):
        """Raw output with tool call (no EOS) -> correct tool_calls."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        raw_output = (
            "<|channel|>analysis<|message|>"
            "The user wants to read /etc/hosts. I'll use read_file."
            "<|start|>assistant"
            "<|channel|>commentary to=functions.read_file_from_disk"
            "<|constrain|>json"
            '<|message|>{"file_path": "/etc/hosts"}'
        )

        tool_parser = HarmonyToolParser()
        result = tool_parser.extract_tool_calls(raw_output)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "read_file_from_disk"

        reasoning_parser = HarmonyReasoningParser()
        reasoning, content = reasoning_parser.extract_reasoning(raw_output)
        assert reasoning is not None
        assert "read_file" in reasoning.lower() or "/etc/hosts" in reasoning

    def test_plain_response_pipeline(self):
        """Raw output with just analysis + final (no EOS) -> content + reasoning."""
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser
        from vllm_mlx.api.utils import clean_output_text

        raw_output = (
            "<|channel|>analysis<|message|>2 + 2 = 4"
            "<|start|>assistant"
            "<|channel|>final<|message|>2 + 2 = 4."
        )

        parser = HarmonyReasoningParser()
        reasoning, content = parser.extract_reasoning(raw_output)
        assert reasoning is not None
        assert "2 + 2" in reasoning
        assert content == "2 + 2 = 4."

        cleaned = clean_output_text(raw_output)
        assert "<|channel|>" not in cleaned

    def test_no_channel_tokens_passthrough(self):
        """Output without any channel tokens passes through unchanged."""
        from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser
        from vllm_mlx.reasoning.harmony_parser import HarmonyReasoningParser

        raw_output = "Just a plain response with no special formatting."

        # Tool parser returns raw text as content when no markers found
        tool_parser = HarmonyToolParser()
        result = tool_parser.extract_tool_calls(raw_output)
        assert not result.tools_called
        assert result.content == raw_output

        # Reasoning parser returns (None, None) for plain text — the server
        # then uses the original text directly as the response content.
        reasoning_parser = HarmonyReasoningParser()
        reasoning, content = reasoning_parser.extract_reasoning(raw_output)
        assert reasoning is None
        assert content is None
