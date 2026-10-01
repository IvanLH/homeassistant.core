"""Reproduction script for duplicate GoogleSearchCallStep in Gemini Interactions API.

Issue:
When using `google-genai` Interactions API with GoogleSearch enabled (`tools=[interactions.GoogleSearch()]`),
the API stream emits:
1. Dynamic model tool call during reasoning: `GoogleSearchCallStep` (signed) -> `GoogleSearchResultStep` (signed)
2. Thought & text output: `ThoughtStep` -> `ModelOutputStep`
3. Post-response grounding injection: A duplicate `GoogleSearchCallStep` (unsigned, server-generated ID)
   with the identical search queries -> `GoogleSearchResultStep` (unsigned) containing the grounding search suggestions.

Usage:
    export GEMINI_API_KEY="your-api-key"
    python repro.py [model_name] [optional_api_key]
"""
# ruff: noqa: T201, BLE001, D103

import asyncio
import os
import sys

from google import genai
from google.genai import interactions

DEFAULT_MODEL = "gemini-3.1-flash-lite"
QUERY = "give me the last 5 winners of the world cup"


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    api_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print(
            "Error: GEMINI_API_KEY environment variable (or 2nd CLI argument) is required."
        )
        print(f"Usage: {sys.argv[0]} [model_name] [api_key]")
        sys.exit(1)

    print(f"Testing model: {model}")
    print(f"Query: {QUERY}\n")
    client = genai.Client(api_key=api_key)

    request = {
        "model": model,
        "input": QUERY,
        "tools": [interactions.GoogleSearch()],
        "stream": True,
        "store": False,
    }

    try:
        stream = await client.aio.interactions.create(**request)
        step_idx = 0
        async for event in stream:
            match event:
                case interactions.StepStart(step=step):
                    step_type = type(step).__name__
                    step_id = getattr(step, "id", getattr(step, "call_id", None))
                    signature = getattr(step, "signature", None)
                    args = getattr(step, "arguments", None)
                    res = getattr(step, "result", None)
                    print(
                        f"\n[StepStart #{step_idx}] type={step_type}, id={step_id}, has_signature={bool(signature)}"
                    )
                    if args:
                        print(f"  arguments: {args}")
                    if res:
                        print(
                            f"  result items: {len(res) if isinstance(res, list) else res}"
                        )
                    step_idx += 1
                case interactions.StepDelta(delta=delta):
                    delta_type = type(delta).__name__
                    text = getattr(delta, "text", None)
                    if text:
                        print(text, end="", flush=True)
                    elif isinstance(delta, interactions.GoogleSearchCallDelta):
                        queries = (
                            list(delta.arguments.queries)
                            if delta.arguments and delta.arguments.queries is not None
                            else None
                        )
                        print(
                            f"\n  [GoogleSearchCallDelta queries={queries}, has_sig={bool(delta.signature)}]",
                            flush=True,
                        )
                    elif isinstance(delta, interactions.GoogleSearchResultDelta):
                        print(
                            f"\n  [GoogleSearchResultDelta count={len(delta.result) if delta.result else 0}, has_sig={bool(delta.signature)}]",
                            flush=True,
                        )
                    else:
                        print(f" [{delta_type}]", end="", flush=True)
                case interactions.StepStop():
                    print("\n[StepStop]")
                case interactions.InteractionCompletedEvent():
                    print("\n[InteractionCompletedEvent]")
                case _:
                    print(f"\n[{type(event).__name__}]")
    except Exception as err:
        print(f"\nError: {type(err).__name__}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
