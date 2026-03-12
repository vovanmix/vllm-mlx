# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis<|message|>Let me think about this...<|end|>
    <|channel|>final<|message|>The answer is 42.<|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.

Note: <|return|> is an EOS token (200002) and may not appear in the
decoded model output.  The regexes accept end-of-string as a fallback.
"""

import re

from .base import DeltaMessage, ReasoningParser

# Analysis channel blocks: <|channel|>analysis<|message|>...<|end|>
_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)(?:<\|end\|>|(?=<\|channel\|>)|(?=<\|start\|>)|$)",
    re.DOTALL,
)

# Final channel content: <|channel|>final<|message|>...<|return|>
# <|return|> is EOS and may be absent from decoded text.
_FINAL_PATTERN = re.compile(
    r"<\|channel\|>final\s*(?:<\|constrain\|>[^<]*)?\s*<\|message\|>(.*?)(?:<\|return\|>|$)",
    re.DOTALL,
)


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Commentary channels (tool calls) are ignored
    since they are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."
    """

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel blocks as reasoning and the
        final channel block as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        # Collect all analysis blocks
        analysis_blocks = _ANALYSIS_PATTERN.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks) or None

        # Extract final channel content
        final_match = _FINAL_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Tracks the current channel and emits reasoning deltas for
        analysis channel content and content deltas for final channel.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: The new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None.
        """
        # 1. Robustly detect active channel from full context
        if "<|channel|>" in current_text:
            last_channel_idx = current_text.rfind("<|channel|>")
            after_channel = current_text[last_channel_idx + len("<|channel|>") :]
            if after_channel.startswith("analysis"):
                self._current_channel = "analysis"
            elif after_channel.startswith("final"):
                self._current_channel = "final"
            elif after_channel.startswith("commentary"):
                self._current_channel = "commentary"

        # 2. Robustly detect if we are currently inside a message block
        self._in_message = False
        if self._current_channel:
            last_channel_idx = current_text.rfind("<|channel|>")
            msg_idx = current_text.find("<|message|>", last_channel_idx)

            if msg_idx != -1:
                after_msg = current_text[msg_idx + len("<|message|>") :]
                # If no terminator has appeared since <|message|> started, we are in the message
                if not any(
                    t in after_msg
                    for t in ("<|end|>", "<|return|>", "<|call|>", "<|start|>")
                ):
                    self._in_message = True

        # Clean any partial/full control tokens out of the delta
        cleaned_delta = re.sub(r"<\|.*?\|>", "", delta_text)

        # Emit content based on current channel
        if self._in_message and cleaned_delta:
            if self._current_channel == "analysis":
                return DeltaMessage(reasoning=cleaned_delta)
            elif self._current_channel == "final":
                return DeltaMessage(content=cleaned_delta)

        # In commentary or unknown channel, suppress
        return None

    def reset_state(self):
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
