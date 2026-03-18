from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

@dataclass
class MemoryNode:
    id: str
    content: str
    importance: int = 5
    confidence: int = 5
    type: str = "fact"
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

@dataclass
class HumanHint:
    text: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class TaskMemoryState:
    task_id: str
    objective: str
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    final_result: Optional[str] = None

    # 短期执行流
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    # 长期事实记忆
    long_term_nodes: List[MemoryNode] = field(default_factory=list)
    # 失败记录与人类提示
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    human_hints: List[HumanHint] = field(default_factory=list)
    # CTF特定状态
    visited_urls: List[str] = field(default_factory=list)

class TaskMemory:
    def __init__(self, task_id: str, memory_dir: str = "./memory"):
        self.task_id = task_id
        self.memory_dir = os.path.abspath(memory_dir)
        _ensure_dir(self.memory_dir)
        self.task_file = os.path.join(self.memory_dir, f"{task_id}.json")
        self.state: Optional[TaskMemoryState] = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.task_file):
            with open(self.task_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.state = TaskMemoryState(
                task_id=raw["task_id"],
                objective=raw.get("objective", raw.get("task_description", "")),
                status=raw.get("status", "running"),
                created_at=raw.get("created_at", time.time()),
                completed_at=raw.get("completed_at"),
                final_result=raw.get("final_result"),
                rounds=raw.get("rounds", []),
                long_term_nodes=[MemoryNode(**n) for n in raw.get("long_term_nodes", [])],
                failed_attempts=raw.get("failed_attempts", []),
                human_hints=[HumanHint(**h) for h in raw.get("human_hints", [])],
                visited_urls=raw.get("visited_urls", [])
            )
        else:
            self.state = TaskMemoryState(task_id=self.task_id, objective="")
            self._save()

    def _save(self) -> None:
        if not self.state:
            return
        with open(self.task_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)

    def set_task_description(self, desc: str) -> None:
        self.state.objective = desc
        self._save()

    def add_round(self, round_idx: int, llm_request: Dict[str, Any], llm_response: Dict[str, Any]) -> None:
        self.state.rounds.append({
            "round": round_idx,
            "timestamp": time.time(),
            "llm_request": llm_request,
            "llm_response": llm_response,
        })
        if len(self.state.rounds) % 3 == 0:
            self._apply_forgetting()
        self._save()

    def add_tool_result(self, round_idx: int, tool_name: str, tool_input: Dict[str, Any], result: Dict[str, Any]) -> None:
        for r in reversed(self.state.rounds):
            if r["round"] == round_idx:
                if "tool_results" not in r:
                    r["tool_results"] = []
                r["tool_results"].append({
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "result": result,
                    "timestamp": time.time(),
                })
                break
        self._save()

    def add_failed_attempt(self, description: str, reason: str) -> None:
        self.state.failed_attempts.append({
            "description": description,
            "reason": reason,
            "timestamp": time.time(),
        })
        self._save()

    def add_human_hint(self, hint: str) -> None:
        self.state.human_hints.append(HumanHint(text=hint, timestamp=time.time()))
        self._save()

    def mark_completed(self, success: bool, final_result: Optional[str] = None) -> None:
        self.state.status = "completed" if success else "failed"
        self.state.completed_at = time.time()
        if final_result:
            self.state.final_result = final_result
        self._save()

    def get_full_context(self) -> Dict[str, Any]:
        return asdict(self.state)

    def store_memory(self, content: str, importance: int, confidence: int, type_: str = "fact") -> str:
        for node in self.state.long_term_nodes:
            if node.content.strip() == content.strip():
                node.importance = max(node.importance, importance)
                node.confidence = max(node.confidence, confidence)
                node.last_accessed = time.time()
                node.access_count += 1
                self._save()
                return node.id

        new_id = str(uuid.uuid4())
        node = MemoryNode(id=new_id, content=content, importance=importance, confidence=confidence, type=type_)
        self.state.long_term_nodes.append(node)
        self._save()
        return node.id

    def retrieve_memory(self, query: str, top_k: int = 3) -> List[MemoryNode]:
        if not self.state.long_term_nodes:
            return []

        current_time = time.time()
        query_words = set(query.lower().split())
        scored_nodes = []

        for node in self.state.long_term_nodes:
            node_words = set(node.content.lower().split())
            intersection = len(query_words.intersection(node_words))
            relevance_score = intersection / max(1, len(query_words))
            importance_score = node.importance / 10.0

            hours_passed = (current_time - node.last_accessed) / 3600.0
            recency_score = math.exp(-0.2 * hours_passed)

            final_score = (relevance_score * 0.5) + (importance_score * 0.3) + (recency_score * 0.2)
            scored_nodes.append((final_score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        results = [node for score, node in scored_nodes[:top_k] if score > 0.1]

        for node in results:
            node.last_accessed = current_time
            node.access_count += 1

        self._save()
        return results

    def _apply_forgetting(self) -> None:
        current_time = time.time()
        survivors = []
        for node in self.state.long_term_nodes:
            hours_passed = (current_time - node.last_accessed) / 3600.0
            retention_factor = math.exp(-0.05 * hours_passed * (11 - node.importance))
            effective_score = node.importance * retention_factor
            if effective_score >= 2.0 or node.importance >= 9:
                survivors.append(node)
        self.state.long_term_nodes = survivors

    def get_working_memory_summary(self, max_recent_rounds: int = 3, query: str = "") -> str:
        lines = [f"Task Objective: {self.state.objective[:300]}"]

        if self.state.human_hints:
            lines.append("\n[CRITICAL: Human Operator Hints]:")
            for hint in self.state.human_hints:
                lines.append(f"  -> {hint.text}")

        if self.state.long_term_nodes:
            retrieved = self.retrieve_memory(query or "CTF flags vulnerability", top_k=4)
            if retrieved:
                lines.append("\n[Retrieved Learned Facts (Long-Term Memory)]:")
                for n in retrieved:
                    lines.append(f"  - [Imp:{n.importance}/10] {n.content}")

        if self.state.failed_attempts:
            lines.append("\n[Recent Failed Tool Attempts (Do not repeat)]:")
            for attempt in self.state.failed_attempts[-3:]:
                lines.append(f"  - {attempt['description']} => {attempt['reason']}")

        recent_rounds = self.state.rounds[-max_recent_rounds:]
        if recent_rounds:
            lines.append(f"\n[Short-Term Action Buffer ({len(recent_rounds)} rounds)]:")
            for r in recent_rounds:
                tools_used = [tr["tool_name"] for tr in r.get("tool_results", [])]
                lines.append(f"  Round {r['round']}: Executed tools={tools_used}")

        return "\n".join(lines)

class TaskMemoryManager:
    def __init__(self, memory_dir: str = "./memory") -> None:
        self.memory_dir = os.path.abspath(memory_dir)
        _ensure_dir(self.memory_dir)

    def create_task(self, task_description: str) -> TaskMemory:
        task_id = f"task_{int(time.time())}_{os.urandom(4).hex()}"
        memory = TaskMemory(task_id, str(self.memory_dir))
        memory.set_task_description(task_description)
        return memory

    def load_task(self, task_id: str) -> TaskMemory:
        return TaskMemory(task_id, str(self.memory_dir))

    def list_tasks(self) -> List[Dict[str, Any]]:
        tasks = []
        for f in Path(self.memory_dir).glob("task_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    tasks.append({
                        "task_id": data["task_id"],
                        "status": data.get("status", "unknown"),
                        "created_at": data.get("created_at"),
                        "description": data.get("objective", "")[:100],
                    })
            except Exception:
                pass
        return sorted(tasks, key=lambda x: x.get("created_at", 0), reverse=True)