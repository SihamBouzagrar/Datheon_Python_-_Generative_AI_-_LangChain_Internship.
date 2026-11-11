"""Task 3: Autonomous Multi-Tool AI Agent.

Architecture:
User Query -> Gemini -> Function Calling
-> Tool Execution -> Observation -> Final Answer

Console output shows the reasoning chain:
THOUGHT -> ACTION -> OBSERVATION -> FINAL ANSWER
"""

import argparse
import os
import sys
import re, time
from datetime import date
from pathlib import Path

# A stale CURL_CA_BUNDLE (e.g. left by a PostgreSQL install) makes the
# search library print a warning on every request. Drop it if the file
# does not exist; system certificates are used instead.
_ca_bundle = os.environ.get("CURL_CA_BUNDLE")
if _ca_bundle and not Path(_ca_bundle).exists():
    os.environ.pop("CURL_CA_BUNDLE", None)

from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_google_genai import ChatGoogleGenerativeAI

from .tools import TOOLS


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Tried in order: if one model is overloaded (503/504), the next is used.
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]

MAX_STEPS = 8


# ---------- System Prompt ----------

def build_system_prompt() -> str:
    return f"""
You are an autonomous AI assistant. Today's date is {date.today().isoformat()}.

You have three tools:
1. web_search(query, recent): search the web (DuckDuckGo).
2. calculator(expression): accurate math.
3. read_local_file(path): read files in the local workspace.

Rules:
- Analyze the question and break compound questions into smaller steps.
- BEFORE EVERY tool call, write one short sentence explaining what you
  are about to do and why (this is your visible reasoning).
- Use the calculator for EVERY calculation, even simple ones.
- If you do not know a file name, call read_local_file with '.' first
  to list the available files.
- For current or latest questions (who holds a role now, news, prices,
  latest versions), call web_search with recent=True and include the
  current year in the query. Prefer news sources over old encyclopedia
  pages, and if results look outdated or conflict, search again with a
  different query before answering.
- State the date of the sources you rely on when the topic is time-sensitive.
- Never invent facts, calculations, or tool results.
- If a tool fails, explain the problem and try another approach.
- Answer in the same language as the user's question.
- Finish with a clear, structured final answer.
"""


# ---------- Helper Function ----------

def extract_text(content) -> str:
    """Extract readable text from Gemini response content."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))

        return "".join(parts).strip()

    return str(content).strip()


# ---------- Agent Execution Loop ----------

def run_agent(
    question: str,
    llm_with_tools,
    verbose: bool = True,
) -> str:

    tools_by_name = {tool.name: tool for tool in TOOLS}

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=question),
    ]

    for step in range(1, MAX_STEPS + 1):

        if verbose:
            print(f"\n{'=' * 55}")
            print(f"AGENT STEP {step}")
            print(f"{'=' * 55}")

        # Ask Gemini what to do next
        try:
            ai_message = invoke_with_retry(llm_with_tools, messages)

        except Exception as exc:
            error = f"LLM execution error: {exc}"

            if verbose:
                print(f"\nERROR: {error}")

            return error

        messages.append(ai_message)

        thought = extract_text(ai_message.content)

        # Final answer: no more tool calls
        if not ai_message.tool_calls:

            if verbose:
                print("\nFINAL ANSWER")
                print("-" * 55)
                print(thought)
                print()

            return thought

        # Visible reasoning before the action
        if verbose and thought:
            print("\nTHOUGHT:")
            print(thought)

        # Execute requested tools
        for call in ai_message.tool_calls:

            tool_name = call["name"]
            tool_args = call["args"]
            tool_call_id = call["id"]

            if verbose:
                print(f"\nACTION: {tool_name}")
                print(f"Arguments: {tool_args}")

            selected_tool = tools_by_name.get(tool_name)

            try:
                if selected_tool:
                    result = selected_tool.invoke(tool_args)
                else:
                    result = f"Error: unknown tool '{tool_name}'."

            except Exception as exc:
                result = f"Tool execution error: {exc}"

            # Display observation
            if verbose:
                preview = str(result).replace("\n", " ")

                print("\nOBSERVATION:")
                print(preview[:500])

            # Send observation back to Gemini
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

    # Safety limit
    message = (
        "The agent stopped because the maximum number "
        "of execution steps was reached."
    )

    if verbose:
        print(f"\n{message}")

    return message


# ---------- Agent Initialization ----------

def build_agent():

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        sys.exit(
            "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY is missing from .env"
        )

    def make(model_name: str):
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key,
            timeout=60,
            max_retries=2,
        )
        return llm.bind_tools(TOOLS)

    primary = make(GEMINI_MODELS[0])
    fallbacks = [make(name) for name in GEMINI_MODELS[1:]]

    # Same .invoke(messages) interface; tries the next model on errors.
    return primary.with_fallbacks(fallbacks)

def invoke_with_retry(llm, messages, max_retries=3):
    """Invoke the LLM; on a 429 quota error, wait the suggested delay."""
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            text = str(exc)
            is_quota = "429" in text or "RESOURCE_EXHAUSTED" in text
            if is_quota and attempt < max_retries:
                match = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
                wait = float(match.group(1)) + 1 if match else 15
                print(f"\n[Quota limit reached - waiting {wait:.0f}s, then retrying...]")
                time.sleep(wait)
                continue
            raise
# ---------- Main Application ----------

def main():

    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Tool AI Agent"
    )

    parser.add_argument(
        "--question",
        "-q",
        help="Ask one question and exit.",
    )

    args = parser.parse_args()

    agent = build_agent()

    # Single-question mode
    if args.question:
        run_agent(args.question, agent)
        return

    # Interactive console mode
    print("\n" + "=" * 55)
    print("MULTI-TOOL AUTONOMOUS AI AGENT")
    print("=" * 55)

    print("\nAvailable tools:")

    for tool in TOOLS:
        first_line = tool.description.strip().splitlines()[0]
        print(f"- {tool.name}: {first_line[:100]}")

    print("\nAgent ready!")
    print("Type 'q', 'quit' or 'exit' to stop.\n")

    while True:

        try:
            question = input("\nUSER: ").strip()

            if question.lower() in {"q", "quit", "exit"}:
                print("\nGoodbye!")
                break

            if not question:
                continue

            run_agent(question, agent)

        except KeyboardInterrupt:
            print("\n\nAgent stopped by user.")
            break


# ---------- Entry Point ----------

if __name__ == "__main__":
    main()