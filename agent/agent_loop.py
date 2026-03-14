from __future__ import annotations

import asyncio
import json
import os
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
""".strip()


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
        os.makedirs(self.runs_dir, exist_ok=True)

    def _new_run_log(self) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.runs_dir, f"run-{ts}.jsonl")
        return path

    def _log_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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

                self._log_jsonl(
                    run_log, {"event": "llm_request", "round": round_idx, "effort": effort}
                )

                resp = await asyncio.to_thread(
                    self.client.create_message,
                    messages=messages,
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

                    for block in content:
                        if block.get("type") != "tool_use":
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

                        if not result.get("ok", False):
                            any_tool_fail = True
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

                    if any_tool_fail:
                        tool_result_blocks.append(
                            {
                                "type": "text",
                                "text": (
                                    "One or more tool calls failed. "
                                    "Analyze the error and try a different approach. "
                                    "Do not repeat the exact same failed request unchanged."
                                ),
                            }
                        )

                    messages.append({"role": "user", "content": tool_result_blocks})
                    continue

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
            }