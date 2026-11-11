"""Task 4: Interactive Streamlit chat app around the Task 3 multi-tool agent.

Features
- st.chat_message / st.chat_input chat interface
- Token-by-token streaming of the agent's answer
- Conversation memory kept in st.session_state (follow-up questions work)
- Visible reasoning chain per answer: thought -> action -> observation
- "New conversation" button for clean session-state management

Run from the project root (the folder that contains `agent_assistant`):
    streamlit run streamlit_app.py
"""

import json
import os
import re
import time

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_assistant.agent import (
    GEMINI_MODELS,
    MAX_STEPS,
    build_agent,
    build_system_prompt,
)
from agent_assistant.tools import TOOLS

st.set_page_config(page_title="Multi-Tool AI Agent", page_icon="🤖")


# ---------- Configuration ----------

MEMORY_NOTE = (
    "\nYou can see the earlier messages of this conversation. "
    "Use them to answer follow-up questions.\n"
)

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


# ---------- Session state management ----------

def new_system_message() -> SystemMessage:
    return SystemMessage(content=build_system_prompt() + MEMORY_NOTE)


def init_state() -> None:
    # "messages": full LangChain history sent to the model (the memory)
    if "messages" not in st.session_state:
        st.session_state.messages = [new_system_message()]
    # "chat": what is displayed in the UI (text + reasoning steps)
    if "chat" not in st.session_state:
        st.session_state.chat = []


def reset_conversation() -> None:
    st.session_state.messages = [new_system_message()]
    st.session_state.chat = []


@st.cache_resource(show_spinner="Connecting to Gemini...")
def get_agent():
    return build_agent()


# ---------- Helpers ----------

def chunk_text(content) -> str:
    """Text of one streamed chunk. Never strip: spaces matter between chunks."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))

        return "".join(parts)

    return ""


def stream_with_retry(agent, messages, on_wait, max_retries=3):
    """Stream the model output; on a 429 quota error, wait and retry."""

    for attempt in range(max_retries + 1):
        started = False

        try:
            for chunk in agent.stream(messages):
                started = True
                yield chunk
            return

        except Exception as exc:
            text = str(exc)
            is_quota = "429" in text or "RESOURCE_EXHAUSTED" in text

            if is_quota and not started and attempt < max_retries:
                match = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
                wait = min(float(match.group(1)) + 1, 90) if match else 20
                on_wait(wait)
                time.sleep(wait)
                continue

            raise


def run_tool(name: str, args: dict) -> str:
    tool = TOOLS_BY_NAME.get(name)

    if tool is None:
        return f"Error: unknown tool '{name}'."

    try:
        return str(tool.invoke(args))
    except Exception as exc:
        return f"Tool execution error: {exc}"


def render_step(container, step: dict) -> None:
    kind = step["kind"]

    if kind == "thought":
        container.markdown(f"💭 **Thought:** {step['text']}")

    elif kind == "action":
        container.markdown(f"🛠️ **Action:** `{step['name']}`")
        container.code(json.dumps(step["args"], ensure_ascii=False), language="json")

    elif kind == "observation":
        container.markdown("👁️ **Observation:**")
        container.code(step["text"][:800], language="text")

    elif kind == "notice":
        container.caption(step["text"])


# ---------- One conversation turn (the agent loop, with streaming) ----------

def run_turn(question: str, status_slot, answer_ph, steps: list) -> str:
    agent = get_agent()
    messages = st.session_state.messages
    messages.append(HumanMessage(content=question))

    ui = {"status": None}

    def add_step(step: dict) -> None:
        steps.append(step)

        if ui["status"] is None:
            ui["status"] = status_slot.status("Agent reasoning", expanded=True)

        render_step(ui["status"], step)

    def on_wait(seconds: float) -> None:
        add_step({
            "kind": "notice",
            "text": f"Free-tier quota reached. Waiting {seconds:.0f}s, then retrying...",
        })

    final_text = ""

    try:
        for _ in range(MAX_STEPS):
            full = None
            text_so_far = ""

            # Stream this step token by token
            for chunk in stream_with_retry(agent, messages, on_wait):
                full = chunk if full is None else full + chunk
                text_so_far += chunk_text(chunk.content)
                answer_ph.markdown(text_so_far + "▌")

            if full is None:
                raise RuntimeError("The model returned an empty response.")

            messages.append(full)

            # No tool call: this streamed text is the final answer
            if not full.tool_calls:
                final_text = text_so_far.strip()
                answer_ph.markdown(final_text)
                break

            # Tool call: the streamed text was the agent's thought
            thought = text_so_far.strip()
            answer_ph.empty()

            if thought:
                add_step({"kind": "thought", "text": thought})

            for call in full.tool_calls:
                add_step({
                    "kind": "action",
                    "name": call["name"],
                    "args": call["args"],
                })

                result = run_tool(call["name"], call["args"])
                add_step({"kind": "observation", "text": result})

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
        else:
            final_text = (
                "The agent stopped because the maximum number of "
                "steps was reached."
            )
            answer_ph.markdown(final_text)

        if ui["status"] is not None:
            tool_calls = sum(1 for s in steps if s["kind"] == "action")
            ui["status"].update(
                label=f"Agent reasoning ({tool_calls} tool call(s))",
                state="complete",
                expanded=False,
            )

        return final_text

    except Exception:
        if ui["status"] is not None:
            ui["status"].update(state="error", expanded=False)
        raise


# ---------- UI ----------

init_state()

with st.sidebar:
    st.header("🤖 Multi-Tool Agent")
    st.caption("Model: " + " → ".join(GEMINI_MODELS))
    st.markdown("**Tools**")

    for tool in TOOLS:
        first_line = tool.description.strip().splitlines()[0]
        st.markdown(f"- `{tool.name}`: {first_line[:70]}")

    st.divider()
    user_turns = sum(1 for m in st.session_state.chat if m["role"] == "user")
    st.metric("Questions in this conversation", user_turns)
    st.button(
        "🗑️ New conversation",
        on_click=reset_conversation,
        use_container_width=True,
    )

st.title("Multi-Tool AI Agent")
st.caption("Ask compound questions: web search, calculator and local files.")

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.error("GEMINI_API_KEY or GOOGLE_API_KEY is missing from your .env file.")
    st.stop()

# Replay the stored conversation
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        if item.get("steps"):
            with st.expander(f"Agent reasoning ({len(item['steps'])} steps)"):
                for step in item["steps"]:
                    render_step(st, step)
        st.markdown(item["content"])

# New question
question = st.chat_input("Ask me anything...")

if question:
    st.session_state.chat.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_slot = st.container()
        answer_ph = st.empty()
        steps: list = []
        history_start = len(st.session_state.messages)

        try:
            answer = run_turn(question, status_slot, answer_ph, steps)

        except Exception as exc:
            # Roll back the half-finished turn so the memory stays valid
            del st.session_state.messages[history_start:]
            answer = f"⚠️ Error: {exc}"
            answer_ph.markdown(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )