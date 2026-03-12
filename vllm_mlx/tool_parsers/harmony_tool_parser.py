# SPDX-License-Identifier: Apache-2.0
"""
Harmony tool call parser for GPT-OSS models.

Harmony uses control tokens and channels for tool calling.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# The model generates: <|channel|>commentary to=functions.NAME json<|message|>ARGS
# <|call|> is an EOS token (200012) and may be absent from decoded text.
# We use a robust regex that handles variable token order and missing EOS tokens.
_COMMENTARY_BLOCK_PATTERN = re.compile(
    r"to=functions\.(\w+).*?<\|message\|>(.*?)(?:<\|call\|>|<\|channel\|>|$)",
    re.DOTALL,
)

# <|return|> is an EOS token and may be absent from decoded text.
_FINAL_BLOCK_PATTERN = re.compile(
    r"<\|channel\|>final.*?<\|message\|>(.*?)(?:<\|return\|>|<\|channel\|>|$)",
    re.DOTALL,
)


@ToolParserManager.register_module(["harmony", "gpt-oss"])
class HarmonyToolParser(ToolParser):
    """
    Tool call parser for GPT-OSS models using Harmony format.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = False

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Harmony model response.

        Parses commentary channel blocks for tool calls and the final
        channel for the user-facing content.
        """
        tool_calls = []

        # Extract tool calls from commentary channel blocks
        for match in _COMMENTARY_BLOCK_PATTERN.finditer(model_output):
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                # Keep the raw arguments string
                tool_calls.append(
                    {
                        "id": _generate_tool_id(),
                        "name": tool_name,
                        "arguments": args_str,
                    }
                )

        # Extract final channel content
        final_match = _FINAL_BLOCK_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        # No tool calls: return all text as content
        # If there's a final channel, use that; otherwise return the raw output
        # stripped of control tokens
        if content is None:
            content = _strip_control_tokens(model_output)

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming Harmony model output.
        """
        if "<|call|>" in delta_text:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        # Find active channel by looking at last <|channel|>
        last_channel_idx = current_text.rfind("<|channel|>")
        if last_channel_idx >= 0:
            active_block = current_text[last_channel_idx:]
            if active_block.startswith("<|channel|>final"):
                if "<|message|>" in active_block:
                    cleaned = delta_text
                    for t in [
                        "<|start|>",
                        "<|end|>",
                        "<|message|>",
                        "<|channel|>",
                        "<|constrain|>",
                        "<|return|>",
                        "<|call|>",
                    ]:
                        cleaned = cleaned.replace(t, "")
                    if cleaned:
                        return {"content": cleaned}
                return None  # Before message or control token
            else:
                return None  # Suppress commentary or analysis

        # No channel marker found yet. Suppress if tool start is detected
        if "to=functions." in current_text:
            return None

        # Clean control tokens and pass through
        cleaned = delta_text
        for t in [
            "<|start|>",
            "<|end|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|return|>",
            "<|call|>",
        ]:
            cleaned = cleaned.replace(t, "")

        if cleaned:
            return {"content": cleaned}

        return None


def _strip_control_tokens(text: str) -> str:
    """Remove Harmony control tokens from text."""
    tokens = [
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|channel|>",
        "<|constrain|>",
        "<|return|>",
        "<|call|>",
    ]
    result = text
    for token in tokens:
        result = result.replace(token, "")
    # Clean up channel names and constrain values
    result = re.sub(r"(?:analysis|commentary|final)\s*", "", result)
    result = re.sub(r"to=functions\.\w+\s*", "", result)
    result = re.sub(r"json\s*", "", result)
    return result.strip()
