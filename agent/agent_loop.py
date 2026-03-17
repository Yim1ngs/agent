from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.task_memory import TaskMemory


WEB_SYSTEM_PROMPT = """
You are a Web CTF solving agent for AUTHORIZED local labs/CTF challenges only.

Operating rules:
1) Prefer tools over guessing.
2) Work step-by-step: request -> inspect -> assert -> iterate.
3) Use extract_artifacts/assert_http after HTTP requests instead of reading huge HTML directly.
4) When you find a candidate flag, write it to the specified file and run pytest verification.
5) If a tool fails or assertions fail, analyze the error and try a different approach.
6) Do NOT repeat the exact same failed request unchanged more than once.
7) Keep outputs concise and actionable.
8) You have access to wsl_shell for running penetration testing tools (nmap, sqlmap, gobuster, etc.) inside WSL Kali Linux.
9) You have access to python_sandbox for executing Python exploit scripts locally.
10) You have a powerful CTF knowledge base. When you identify a vulnerability direction (e.g., SSTI, SQLi, ThinkPHP), use search_knowledge FIRST before blindly testing.
""".strip()

# Phase 2 config
CONTEXT_WINDOW_MAX_MESSAGES = 16  # Keep first message (system) + last N messages
CONSECUTIVE_FAIL_THRESHOLD = 3   # Trigger advisor after N rounds without flag


class WebCTFAgent:
    def __init__(
            self,
            client,
            tools,
            runs_dir: str = "./runs",
            max_rounds: int = 20,
            memory: Optional[TaskMemory] = None,
            mcp_configs: Optional[List[StdioServerParameters]] = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.runs_dir = runs_dir
        self.max_rounds = max_rounds
        self.memory = memory
        self.mcp_configs = mcp_configs or []
        self._consecutive_no_flag_rounds = 0
        self._advisor_called_count = 0
        os.makedirs(self.runs_dir, exist_ok=True)

    def _new_run_log(self) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.runs_dir, f"run-{ts}.jsonl")
        return path

    def _log_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # =========================================
    # Phase 2: Context Sliding Window
    # =========================================
    @staticmethod
    def _apply_sliding_window(messages: List[Dict[str, Any]], max_messages: int = CONTEXT_WINDOW_MAX_MESSAGES) -> List[Dict[str, Any]]:
        """Keep the first message (system/task prompt) and the most recent messages.
        This prevents token exhaustion and 'middle forgetting' in the LLM."""
        if len(messages) <= max_messages:
            return messages

        first_msg = messages[0]
        recent = messages[-(max_messages - 1):]

        # Ensure the window starts with a user message (API requirement)
        if recent and recent[0]["role"] == "assistant":
            recent = recent[1:]

        return [first_msg] + recent

    # =========================================
    # Phase 2: Dynamic Memory Injection
    # =========================================
    def _inject_memory_update(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject the latest TaskMemory summary into the last user message."""
        if not self.memory:
            return messages

        summary = self.memory.get_working_memory_summary()
        memory_block = f"\n\n[System Update: Current Memory]\n{summary}"

        if not messages:
            return messages

        msgs = list(messages)
        last = msgs[-1]

        if last["role"] == "user":
            if isinstance(last["content"], str):
                msgs[-1] = {**last, "content": last["content"] + memory_block}
            elif isinstance(last["content"], list):
                msgs[-1] = {
                    **last,
                    "content": last["content"] + [{"type": "text", "text": memory_block}]
                }

        return msgs

    # =========================================
    # Phase 3: Advisor Agent
    # =========================================
    def _should_consult_advisor(self) -> bool:
        """Check if the attacker has been stuck and needs strategic advice."""
        return self._consecutive_no_flag_rounds >= CONSECUTIVE_FAIL_THRESHOLD

    async def _consult_advisor(self, task: str, run_log: str) -> Optional[str]:
        """Call a separate LLM as a strategic Advisor with a clean, focused prompt."""
        if not self.memory:
            return None

        self._advisor_called_count += 1
        summary = self.memory.get_working_memory_summary(max_recent_rounds=5)

        recent_failures = []
        for attempt in self.memory.state.failed_attempts[-5:]:
            recent_failures.append(f"  - {attempt['description']} => {attempt['reason']}")
        failures_text = "\n".join(recent_failures) if recent_failures else "  (no recorded failures)"

        advisor_prompt = f"""You are a senior CTF security consultant (Advisor).
An automated attacker agent has been working on a CTF challenge but is stuck and making no progress.

Your job: Analyze the situation and provide HIGH-LEVEL strategic guidance.
Do NOT write tool calls or code. Just provide clear, actionable advice in plain text.

=== CHALLENGE OBJECTIVE ===
{task[:500]}

=== ATTACKER'S CURRENT STATE (Memory Summary) ===
{summary}

=== RECENT FAILED ATTEMPTS ===
{failures_text}

=== YOUR TASK ===
Based on the above, provide:
1. What the attacker might be doing wrong or missing
2. Alternative attack vectors to explore (be specific: mention tool names, techniques, payloads)
3. A concrete next-step recommendation (e.g., "Try SSTI with {{{{7*7}}}} on the /search endpoint")

Be concise, direct, and actionable. Max 300 words."""

        try:
            advisor_resp = await asyncio.to_thread(
                self.client.create_message,
                messages=[{"role": "user", "content": advisor_prompt}],
                tools=None,
                max_tokens=800,
                effort="high",
                stream=False,
            )

            advice_parts = []
            for block in advisor_resp.get("content", []):
                if block.get("type") == "text":
                    advice_parts.append(block.get("text", ""))
            advice = "\n".join(advice_parts).strip()

            self._log_jsonl(run_log, {
                "event": "advisor_consultation",
                "advice": advice,
                "advisor_call_count": self._advisor_called_count,
            })

            return advice
        except Exception as e:
            self._log_jsonl(run_log, {
                "event": "advisor_error",
                "error": str(e),
            })
            return None

    # =========================================
    # Phase 3+: Context Reset on Advisor Trigger
    # =========================================
    def _compress_and_reset_context(
        self, task: str, advice: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """When the advisor triggers, compress all prior history into a concise
        briefing and return a fresh message list.  This prevents the attacker
        from being anchored by verbose failure outputs."""

        # --- 1. Build compressed summary of what happened so far ---
        sections: List[str] = []

        # Discovered facts from long-term memory
        if self.memory and self.memory.state.long_term_nodes:
            facts = []
            for node in sorted(
                self.memory.state.long_term_nodes,
                key=lambda n: n.importance,
                reverse=True,
            )[:8]:
                facts.append(f"  - [Imp:{node.importance}] {node.content}")
            if facts:
                sections.append("[Confirmed Discoveries]\n" + "\n".join(facts))

        # Failed approaches — deduplicated, concise
        if self.memory and self.memory.state.failed_attempts:
            seen = set()
            fail_lines = []
            for attempt in self.memory.state.failed_attempts:
                key = attempt["description"][:80]
                if key not in seen:
                    seen.add(key)
                    fail_lines.append(
                        f"  - {attempt['description'][:120]} => {attempt['reason'][:100]}"
                    )
            # Keep at most 10 unique failures
            if fail_lines:
                sections.append(
                    "[Approaches Already Tried and FAILED — Do NOT repeat these]\n"
                    + "\n".join(fail_lines[-10:])
                )

        # Previous human/advisor hints
        if self.memory and self.memory.state.human_hints:
            hint_lines = [f"  - {h.text[:200]}" for h in self.memory.state.human_hints[-5:]]
            sections.append("[Previous Hints & Advice]\n" + "\n".join(hint_lines))

        # Visited URLs
        if self.memory and self.memory.state.visited_urls:
            sections.append(
                "[Known URLs]\n  " + ", ".join(self.memory.state.visited_urls[-10:])
            )

        compressed_history = "\n\n".join(sections) if sections else "(no prior context)"

        # --- 2. Build the reset message ---
        system_context = WEB_SYSTEM_PROMPT
        if self.memory:
            memory_summary = self.memory.get_working_memory_summary()
            system_context += f"\n\n=== Task Memory ===\n{memory_summary}\n"

        reset_user_msg = (
            f"{system_context}\n\n"
            f"Task:\n{task}\n\n"
            f"{'=' * 60}\n"
            f"[CONTEXT RESET — Advisor Consultation #{self._advisor_called_count}]\n"
            f"The previous attack phase did NOT find the flag after multiple rounds.\n"
            f"Below is a compressed summary of everything tried so far, followed by\n"
            f"strategic advice from a senior CTF consultant.\n"
            f"{'=' * 60}\n\n"
            f"## Prior Work Summary\n{compressed_history}\n\n"
            f"## Advisor Strategic Guidance\n{advice}\n\n"
            f"{'=' * 60}\n"
            f"[ACTION REQUIRED] Based on the advisor's guidance above, take a\n"
            f"FUNDAMENTALLY DIFFERENT approach. Do NOT repeat any of the failed\n"
            f"methods listed above. Start with the advisor's concrete recommendation."
        )

        return [{"role": "user", "content": reset_user_msg}]

    # =========================================
    # Phase 2: Check for flag in tool results
    # =========================================
    @staticmethod
    def _check_flag_in_results(tool_results: List[Dict[str, Any]], flag_regex: str = r"flag\{[A-Za-z0-9_\-]+\}") -> bool:
        """Check if any tool result contains a potential flag pattern."""
        for tr in tool_results:
            result_str = json.dumps(tr, ensure_ascii=False)
            if re.search(flag_regex, result_str, re.IGNORECASE):
                return True
        return False

    # =========================================
    # Main solve loop
    # =========================================
    async def solve(
            self, task: str, resume_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        run_log = self._new_run_log()

        system_context = WEB_SYSTEM_PROMPT
        if self.memory:
            memory_summary = self.memory.get_working_memory_summary()
            system_context += f"\n\n=== Task Memory ===\n{memory_summary}\n"

        if resume_messages:
            messages = [
                {
                    "role": "user",
                    "content": f"{system_context}\n\n[Task Resumed - Target URL might have been updated. Use the new URL below!]:\n{task}"
                }
            ]
            messages.extend(resume_messages)
        else:
            messages = [
                {"role": "user", "content": f"{system_context}\n\nTask:\n{task}"}
            ]

        tools_def = self.tools.anthropic_tools()
        mcp_sessions: Dict[str, ClientSession] = {}

        async with AsyncExitStack() as stack:
            if self.mcp_configs:
                for config in self.mcp_configs:
                    read, write = await stack.enter_async_context(stdio_client(config))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()

                    mcp_tools = await session.list_tools()
                    for t in mcp_tools.tools:
                        tools_def.append({
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.inputSchema
                        })
                        mcp_sessions[t.name] = session

            self._log_jsonl(run_log, {"event": "start", "task": task})

            for round_idx in range(1, self.max_rounds + 1):
                effort = "high" if round_idx <= 8 else "max"

                # --- Phase 2: Apply sliding window ---
                windowed_messages = self._apply_sliding_window(messages)

                # --- Phase 2: Inject dynamic memory ---
                windowed_messages = self._inject_memory_update(windowed_messages)

                self._log_jsonl(
                    run_log, {
                        "event": "llm_request",
                        "round": round_idx,
                        "effort": effort,
                        "message_count": len(windowed_messages),
                        "original_message_count": len(messages),
                    }
                )

                resp = await asyncio.to_thread(
                    self.client.create_message,
                    messages=windowed_messages,
                    tools=tools_def,
                    max_tokens=1400,
                    effort=effort,
                    stream=False,
                )

                stop_reason = resp.get("stop_reason")
                content = resp.get("content", [])

                self._log_jsonl(
                    run_log,
                    {
                        "event": "llm_response",
                        "round": round_idx,
                        "stop_reason": stop_reason,
                        "content": content,
                    },
                )

                if self.memory:
                    self.memory.add_round(round_idx, {"effort": effort}, resp)

                messages.append({"role": "assistant", "content": content})

                if stop_reason == "tool_use":
                    tool_result_blocks = []
                    any_tool_fail = False
                    circuit_broken = False
                    round_tool_results = []

                    for block in content:
                        if block.get("type") != "tool_use":
                            continue

                        # --- Phase 2: Circuit Breaker ---
                        # If a previous tool in this round failed, skip remaining tools
                        if circuit_broken:
                            tool_result_blocks.append({
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": json.dumps({
                                    "ok": False,
                                    "error": "CIRCUIT BREAKER: Skipped because a previous tool in this batch failed. Fix the failure first.",
                                }, ensure_ascii=False),
                            })
                            continue

                        tool_name = block["name"]
                        tool_use_id = block["id"]
                        tool_input = block.get("input", {})

                        if tool_name in mcp_sessions:
                            try:
                                mcp_result = await mcp_sessions[tool_name].call_tool(tool_name, arguments=tool_input)
                                res_text = "\n".join([c.text for c in mcp_result.content if c.type == "text"])
                                result = {"ok": True, "mcp_output": res_text}
                            except Exception as e:
                                result = {"ok": False, "error": str(e)}
                        else:
                            result = self.tools.call(tool_name, tool_input)

                        round_tool_results.append(result)

                        if not result.get("ok", False):
                            any_tool_fail = True
                            circuit_broken = True  # Phase 2: Activate circuit breaker
                            if self.memory:
                                self.memory.add_failed_attempt(
                                    f"{tool_name}({json.dumps(tool_input)[:100]})",
                                    result.get("error", "unknown"),
                                )

                        self._log_jsonl(
                            run_log,
                            {
                                "event": "tool_result",
                                "round": round_idx,
                                "tool_name": tool_name,
                                "tool_use_id": tool_use_id,
                                "tool_input": tool_input,
                                "result": result,
                                "circuit_broken": circuit_broken,
                            },
                        )

                        if self.memory:
                            self.memory.add_tool_result(
                                round_idx, tool_name, tool_input, result
                            )

                        tool_result_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )

                    # --- Phase 2: Track flag progress for advisor trigger ---
                    flag_regex = os.getenv("CTF_FLAG_REGEX", r"flag\{[A-Za-z0-9_\-]+\}")
                    if self._check_flag_in_results(round_tool_results, flag_regex):
                        self._consecutive_no_flag_rounds = 0
                    else:
                        self._consecutive_no_flag_rounds += 1

                    if any_tool_fail:
                        tool_result_blocks.append(
                            {
                                "type": "text",
                                "text": (
                                    "[CIRCUIT BREAKER ACTIVATED] One or more tool calls failed. "
                                    "Remaining tools in this batch were skipped. "
                                    "Analyze the error and try a different approach. "
                                    "Do not repeat the exact same failed request unchanged."
                                ),
                            }
                        )

                    # --- Phase 3: Consult Advisor if stuck ---
                    if self._should_consult_advisor():
                        print(f"    [Advisor] Attacker stuck for {self._consecutive_no_flag_rounds} rounds, consulting advisor...")
                        advice = await self._consult_advisor(task, run_log)
                        if advice:
                            if self.memory:
                                self.memory.add_human_hint(f"[Advisor #{self._advisor_called_count}]: {advice[:300]}")

                            # Context Reset: compress history + advisor advice → fresh start
                            messages = self._compress_and_reset_context(task, advice, messages)
                            self._consecutive_no_flag_rounds = 0

                            self._log_jsonl(run_log, {
                                "event": "context_reset",
                                "advisor_call_count": self._advisor_called_count,
                                "new_message_count": len(messages),
                            })
                            continue  # Skip appending tool_result_blocks, restart with clean context

                    messages.append({"role": "user", "content": tool_result_blocks})
                    continue

                # stop_reason != "tool_use" => final text response
                text_parts = []
                for b in content:
                    if b.get("type") == "text":
                        text_parts.append(b.get("text", ""))
                final_text = "\n".join(text_parts).strip()

                self._log_jsonl(
                    run_log,
                    {
                        "event": "final_text",
                        "round": round_idx,
                        "text": final_text,
                    },
                )

                if self.memory:
                    self.memory.mark_completed(True, final_text)

                return {
                    "ok": True,
                    "rounds": round_idx,
                    "response": final_text,
                    "run_log": run_log,
                    "messages": messages,
                    "raw": resp,
                    "advisor_calls": self._advisor_called_count,
                }

            if self.memory:
                self.memory.mark_completed(
                    False, f"max rounds exceeded ({self.max_rounds})"
                )

            return {
                "ok": False,
                "error": f"max rounds exceeded ({self.max_rounds})",
                "run_log": run_log,
                "messages": messages,
                "advisor_calls": self._advisor_called_count,
            }
