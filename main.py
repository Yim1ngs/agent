from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from agent.client_claude import ClaudeNewAPIClient
from agent.web_tools import WebToolRegistry
from agent.agent_loop import WebCTFAgent
from agent.task_memory import TaskMemoryManager


def _load_env() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here.parent / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def main() -> None:
    _load_env()

    api_key = _require_env("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "http://newapi.200m.997555.xyz").strip()
    model = os.getenv("LLM_MODEL_ID", "claude-opus-4-6").strip()

    target_url = _require_env("CTF_TARGET_URL").rstrip("/")

    workspace_root = os.getenv("CTF_WORKSPACE_ROOT", "./challenges").strip()
    candidate_flag_path = os.getenv(
        "CTF_CANDIDATE_FLAG_PATH",
        "challenges/web_demo/workspace/candidate_flag.txt",
    ).strip()
    test_path = os.getenv(
        "CTF_TEST_PATH", "challenges/web_demo/tests/test_success.py"
    ).strip()
    test_cwd = os.getenv("CTF_TEST_CWD", "challenges").strip()

    writable_roots_raw = os.getenv("CTF_WRITABLE_ROOTS", "").strip()
    if writable_roots_raw:
        writable_roots = [p.strip() for p in writable_roots_raw.split(",") if p.strip()]
    else:
        writable_roots = [str(Path(candidate_flag_path).parent)]

    allowed_hosts_raw = os.getenv("CTF_ALLOWED_HOSTS", "").strip()
    allowed_hosts = (
        [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()]
        if allowed_hosts_raw
        else None
    )

    flag_regex = os.getenv("CTF_FLAG_REGEX", r"^flag\{[A-Za-z0-9_\-]+\}$").strip()

    memory_dir = os.getenv("MEMORY_DIR", "./memory").strip()
    memory_manager = TaskMemoryManager(memory_dir)

    resume_task_id = os.getenv("RESUME_TASK_ID", "").strip()
    human_hint = os.getenv("HUMAN_HINT", "").strip()

    client = ClaudeNewAPIClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_sec=int(os.getenv("NEWAPI_TIMEOUT_SEC", "90")),
    )

    tools = WebToolRegistry(
        workspace_root=workspace_root,
        allowed_hosts=allowed_hosts,
        writable_roots=writable_roots,
    )

    task_description = f"""
You are solving an authorized ONLINE web CTF challenge.

Target base URL:
- {target_url}

Goals:
1) Explore the website and find the REAL flag.
2) Write the candidate flag ONLY into:
   {candidate_flag_path}
3) Run pytest verification using:
   test_path = {test_path}
   cwd = {test_cwd}
4) If verification fails, continue investigating and retry.
5) Report the final verified flag and the successful path used.

Important operating constraints:
- Do NOT create, modify, or overwrite any test files.
- Tests are trusted and immutable. If a test file is missing, report the issue instead of creating one.
- Use extract_artifacts/assert_http after requests whenever helpful.
- Use validate_candidate_flag before treating any value as a final flag.
- A candidate is only valid if it BOTH:
  (a) matches the expected platform flag regex: {flag_regex}
  (b) passes pytest verification
- If you find a flag-like string that does NOT match the regex (e.g., examples/sample flags such as Syc{{...}}),
  treat it as a clue or artifact, NOT the final answer.
- Avoid repeating the exact same failed request unchanged.
""".strip()

    if resume_task_id:
        print(f"\n=== RESUMING TASK: {resume_task_id} ===")
        memory = memory_manager.load_task(resume_task_id)

        if human_hint:
            print(f"Human hint: {human_hint}")
            memory.add_human_hint(human_hint)

        context = memory.get_full_context()
        last_messages = []
        for r in context.get("rounds", []):
            llm_resp = r.get("llm_response", {})
            content = llm_resp.get("content", [])
            if content:
                last_messages.append({"role": "assistant", "content": content})

            if "tool_results" in r:
                tool_result_blocks = []
                for tr in r["tool_results"]:
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": f"resumed_{r['round']}",
                            "content": str(tr["result"]),
                        }
                    )
                if tool_result_blocks:
                    last_messages.append(
                        {"role": "user", "content": tool_result_blocks}
                    )

        if human_hint:
            last_messages.append(
                {
                    "role": "user",
                    "content": f"Human operator hint: {human_hint}\n\nPlease continue based on this hint and previous attempts.",
                }
            )

        agent = WebCTFAgent(
            client=client,
            tools=tools,
            runs_dir=os.getenv("RUNS_DIR", "./runs"),
            max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "20")),
            memory=memory,
        )

        result = agent.solve(
            task_description, resume_messages=last_messages if last_messages else None
        )
    else:
        print("\n=== STARTING NEW TASK ===")
        memory = memory_manager.create_task(task_description)
        print(f"Task ID: {memory.task_id}")

        agent = WebCTFAgent(
            client=client,
            tools=tools,
            runs_dir=os.getenv("RUNS_DIR", "./runs"),
            max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "20")),
            memory=memory,
        )

        result = agent.solve(task_description)

    print("\n=== AGENT RESULT ===")
    if result.get("ok"):
        print(result.get("response", ""))
    else:
        print("ERROR:", result.get("error"))
        print(f"\nTo resume this task, set: RESUME_TASK_ID={memory.task_id}")
        print("To add human hint, set: HUMAN_HINT='your hint here'")

    print(f"\n[run log] {result.get('run_log')}")
    print(f"[memory] {memory.task_file}")


if __name__ == "__main__":
    main()
