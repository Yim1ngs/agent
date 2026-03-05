from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskMemory:
    def __init__(self, task_id: str, memory_dir: str = "./memory") -> None:
        self.task_id = task_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.task_file = self.memory_dir / f"{task_id}.json"

        if self.task_file.exists():
            with open(self.task_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {
                "task_id": task_id,
                "created_at": time.time(),
                "status": "running",
                "task_description": "",
                "rounds": [],
                "artifacts": {},
                "human_hints": [],
                "verified_facts": {},
                "failed_attempts": [],
            }
            self._save()

    def _save(self) -> None:
        with open(self.task_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def set_task_description(self, desc: str) -> None:
        self.data["task_description"] = desc
        self._save()

    def add_round(
        self, round_idx: int, llm_request: Dict[str, Any], llm_response: Dict[str, Any]
    ) -> None:
        self.data["rounds"].append(
            {
                "round": round_idx,
                "timestamp": time.time(),
                "llm_request": llm_request,
                "llm_response": llm_response,
            }
        )
        self._save()

    def add_tool_result(
        self,
        round_idx: int,
        tool_name: str,
        tool_input: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        for r in reversed(self.data["rounds"]):
            if r["round"] == round_idx:
                if "tool_results" not in r:
                    r["tool_results"] = []
                r["tool_results"].append(
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "result": result,
                        "timestamp": time.time(),
                    }
                )
                break
        self._save()

    def add_artifact(self, key: str, value: Any, confidence: str = "medium") -> None:
        self.data["artifacts"][key] = {
            "value": value,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        self._save()

    def add_human_hint(self, hint: str) -> None:
        self.data["human_hints"].append(
            {
                "hint": hint,
                "timestamp": time.time(),
            }
        )
        self._save()

    def add_verified_fact(self, key: str, value: Any) -> None:
        self.data["verified_facts"][key] = {
            "value": value,
            "timestamp": time.time(),
        }
        self._save()

    def add_failed_attempt(self, description: str, reason: str) -> None:
        self.data["failed_attempts"].append(
            {
                "description": description,
                "reason": reason,
                "timestamp": time.time(),
            }
        )
        self._save()

    def mark_completed(self, success: bool, final_result: Optional[str] = None) -> None:
        self.data["status"] = "completed" if success else "failed"
        self.data["completed_at"] = time.time()
        if final_result:
            self.data["final_result"] = final_result
        self._save()

    def get_working_memory_summary(self, max_recent_rounds: int = 3) -> str:
        lines = [f"Task: {self.data['task_description'][:200]}"]

        if self.data["verified_facts"]:
            lines.append("\nVerified Facts:")
            for k, v in list(self.data["verified_facts"].items())[-5:]:
                lines.append(f"  - {k}: {v['value']}")

        if self.data["artifacts"]:
            lines.append("\nKey Artifacts:")
            for k, v in list(self.data["artifacts"].items())[-5:]:
                lines.append(f"  - {k} ({v['confidence']}): {str(v['value'])[:100]}")

        if self.data["failed_attempts"]:
            lines.append("\nRecent Failed Attempts:")
            for attempt in self.data["failed_attempts"][-3:]:
                lines.append(f"  - {attempt['description']}: {attempt['reason']}")

        if self.data["human_hints"]:
            lines.append("\nHuman Hints:")
            for hint in self.data["human_hints"]:
                lines.append(f"  - {hint['hint']}")

        recent_rounds = self.data["rounds"][-max_recent_rounds:]
        if recent_rounds:
            lines.append(f"\nRecent {len(recent_rounds)} rounds summary:")
            for r in recent_rounds:
                tools_used = []
                if "tool_results" in r:
                    tools_used = [tr["tool_name"] for tr in r["tool_results"]]
                lines.append(f"  Round {r['round']}: tools={tools_used}")

        return "\n".join(lines)

    def get_full_context(self) -> Dict[str, Any]:
        return self.data.copy()


class TaskMemoryManager:
    def __init__(self, memory_dir: str = "./memory") -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, task_description: str) -> TaskMemory:
        task_id = f"task_{int(time.time())}_{os.urandom(4).hex()}"
        memory = TaskMemory(task_id, str(self.memory_dir))
        memory.set_task_description(task_description)
        return memory

    def load_task(self, task_id: str) -> TaskMemory:
        return TaskMemory(task_id, str(self.memory_dir))

    def list_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for f in self.memory_dir.glob("task_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    tasks.append(
                        {
                            "task_id": data["task_id"],
                            "status": data.get("status", "unknown"),
                            "created_at": data.get("created_at"),
                            "description": data.get("task_description", "")[:100],
                        }
                    )
            except Exception:
                pass
        return sorted(tasks, key=lambda x: x.get("created_at", 0), reverse=True)
