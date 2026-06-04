"""Custom Petri target agent that pins the deployed triage SYSTEM_PROMPT.

Petri's `audit()` task has NO parameter to fix the target's system prompt: the
stock `inspect_petri.target.target_agent` uses whatever system message the
AUDITOR stages during scenario setup (`await context.system_message()`). For a
faithful evaluation of the DEPLOYED configuration the target must always run with
our single-source `SYSTEM_PROMPT` (from `static_evaluation/prompts.py`), not an
auditor-invented one.

This is a thin vendored copy of `target_agent` (verified against inspect_petri
`main`) whose only change is: it still CONSUMES the auditor-staged system message
(to keep the audit channel in sync) but DISCARDS it and forces
`ChatMessageSystem(content=SYSTEM_PROMPT)`. No prompt duplication (single-source
import), no model change.

Pass via `audit(target=triage_target_agent(), target_tools="none")`.

Coupling note: this depends on inspect-petri's target internals
(`ExitSignal`, `TOOL_RESULT`). Pin the inspect-petri version; if a future release
changes the target loop, re-sync this copy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import (
    CachePolicy,
    ChatMessageAssistant,
    ChatMessageSystem,
    get_model,
)

from inspect_petri.target._context import ExitSignal, TargetContext
from inspect_petri.target._types import TOOL_RESULT

# Single source of truth for the deployed triage system prompt.
_STATIC_EVAL = Path(__file__).resolve().parent.parent / "static_evaluation"
if str(_STATIC_EVAL) not in sys.path:
    sys.path.insert(0, str(_STATIC_EVAL))
from prompts import SYSTEM_PROMPT  # noqa: E402


@agent(name="triage_target")
def triage_target_agent(*, cache: bool | CachePolicy = False) -> Agent:
    """Target agent that forces the deployed triage SYSTEM_PROMPT.

    Mirrors `inspect_petri.target.target_agent` exactly except the system message
    is pinned to `SYSTEM_PROMPT` instead of the auditor-staged one.
    """

    async def execute(state: AgentState, context: TargetContext,
                      metadata: dict[str, Any]) -> AgentState:
        target_model = get_model(role="target")
        generate = context.replayable(target_model.generate, surface_errors=True)
        target_cache = context.scoped_cache(cache)

        try:
            await context.wait_for_resume()
            # Consume (and discard) the auditor-staged system message so the
            # channel stays in sync, then PIN our deployed system prompt.
            await context.system_message()
            state.messages[:] = [
                ChatMessageSystem(content=SYSTEM_PROMPT),
                await context.user_message(),
            ]

            while True:
                if prefill := context.prefill():
                    state.messages.append(ChatMessageAssistant(content=prefill))

                state.output = await generate(
                    input=state.messages, tools=context.tools, cache=target_cache
                )
                state.messages.append(state.output.message)

                if tool_calls := state.output.message.tool_calls:
                    context.expect({TOOL_RESULT: {tc.id for tc in tool_calls}})
                    await context.send_output(state.output)
                    state.messages.extend(await context.tool_results(tool_calls))
                    if user_message := await context.user_message(required=False):
                        state.messages.append(user_message)
                else:
                    context.expect({TOOL_RESULT: set()})
                    await context.send_output(state.output)
                    state.messages.append(await context.user_message())
        except ExitSignal:
            return state

    return execute
