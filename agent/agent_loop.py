from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

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
    ) -> None:
        self.client = client
        self.tools = tools
        self.runs_dir = runs_dir
        self.max_rounds = max_rounds
        self.memory = memory
        os.makedirs(self.runs_dir, exist_ok=True)

    def _new_run_log(self) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.runs_dir, f"run-{ts}.jsonl")
        return path

    def _log_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def solve(
        self, task: str, resume_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        run_log = self._new_run_log()

        if resume_messages:
            messages = resume_messages
        else:
            system_context = WEB_SYSTEM_PROMPT
            if self.memory:
                memory_summary = self.memory.get_working_memory_summary()
                system_context += f"\n\n=== Task Memory ===\n{memory_summary}\n"

            messages = [
                {"role": "user", "content": f"{system_context}\n\nTask:\n{task}"}
            ]

        tools_def = self.tools.anthropic_tools()

        self._log_jsonl(run_log, {"event": "start", "task": task})

        for round_idx in range(1, self.max_rounds + 1):
            effort = "medium" if round_idx <= 8 else "low"

            self._log_jsonl(
                run_log, {"event": "llm_request", "round": round_idx, "effort": effort}
            )

            resp = self.client.create_message(
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

                messages.append({"role": "user", "content": tool_result_blocks})

                if any_tool_fail:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "One or more tool calls failed. "
                                "Analyze the error and try a different approach. "
                                "Do not repeat the exact same failed request unchanged."
                            ),
                        }
                    )
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
