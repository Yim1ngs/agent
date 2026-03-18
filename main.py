from __future__ import annotations

import os
import asyncio
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

from mcp import StdioServerParameters

from agent.client_claude import ClaudeNewAPIClient
from agent.web_tools import WebToolRegistry
from agent.agent_loop import WebCTFAgent
from agent.task_memory import TaskMemoryManager, TaskMemory

def _load_env() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here.parent / ".env",
        ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)

def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def build_resume_messages(memory: TaskMemory, user_input: str) -> list[dict]:
    """从 TaskMemory 中提取历史轮次，并拼接用户最新的输入"""
    context = memory.get_full_context()
    last_messages = []

    for r in context.get("rounds", []):
        llm_resp = r.get("llm_response", {})
        content = llm_resp.get("content", [])

        if not content:
            continue

        assistant_text = ""
        for block in content:
            if block.get("type") == "text":
                assistant_text += block.get("text", "") + "\n"
            elif block.get("type") == "tool_use":
                assistant_text += f"[Action: Tool '{block.get('name')}' called with input: {json.dumps(block.get('input'), ensure_ascii=False)}]\n"

        if assistant_text:
            last_messages.append({"role": "assistant", "content": assistant_text.strip()})

        if "tool_results" in r:
            user_text = ""
            for tr in r["tool_results"]:
                user_text += f"[Tool Result for '{tr.get('tool_name')}']: {json.dumps(tr.get('result'), ensure_ascii=False)}\n"

            if user_text:
                last_messages.append({"role": "user", "content": user_text.strip()})

    system_warning = (
        f"\n\nCRITICAL SYSTEM WARNING:\n"
        f"In the history above, past tool calls are shown as text `[Action: Tool ...]`. "
        f"This is ONLY a transcript. You CANNOT execute tools by typing `[Action: Tool ...]`. "
        f"You MUST strictly use the native JSON Tool Calling mechanism to make your next move!"
    )

    final_text = f"Human operator input/hint: {user_input}" + system_warning if user_input else "System Resume Notice: Continuing task." + system_warning

    if last_messages and last_messages[-1]["role"] == "user":
        if isinstance(last_messages[-1]["content"], list):
            last_messages[-1]["content"].append({"type": "text", "text": final_text})
        else:
            last_messages[-1]["content"] += f"\n\n{final_text}"
    else:
        last_messages.append({"role": "user", "content": final_text})

    return last_messages

async def main() -> None:
    _load_env()

    api_key = _require_env("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "http://newapi.200m.997555.xyz").strip()
    model = os.getenv("LLM_MODEL_ID", "claude-opus-4-6").strip()

    target_url = os.getenv("CTF_TARGET_URL", "Target URL pending from user").rstrip("/")
    workspace_root = os.getenv("CTF_WORKSPACE_ROOT", "./challenges").strip()
    candidate_flag_path = os.getenv("CTF_CANDIDATE_FLAG_PATH", "challenges/web_demo/workspace/candidate_flag.txt").strip()
    test_path = os.getenv("CTF_TEST_PATH", "challenges/web_demo/tests/test_success.py").strip()
    test_cwd = os.getenv("CTF_TEST_CWD", "challenges").strip()

    writable_roots_raw = os.getenv("CTF_WRITABLE_ROOTS", "").strip()
    if writable_roots_raw:
        writable_roots = [p.strip() for p in writable_roots_raw.split(",") if p.strip()]
    else:
        writable_roots = [str(Path(candidate_flag_path).parent)]

    allowed_hosts_raw = os.getenv("CTF_ALLOWED_HOSTS", "").strip()
    allowed_hosts = [h.strip() for h in allowed_hosts_raw.split(",") if h.strip()] if allowed_hosts_raw else None
    flag_regex = os.getenv("CTF_FLAG_REGEX", r"^flag\{[A-Za-z0-9_\-]+\}$").strip()

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

    mcp_servers = [
        StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-puppeteer"],
        ),
    ]

    memory_dir = os.getenv("MEMORY_DIR", "./memory").strip()
    memory_manager = TaskMemoryManager(memory_dir)

    max_rounds = int(os.getenv("AGENT_MAX_ROUNDS", "5"))

    print("=====================================================")
    print("启动。。。")
    print(f"当前模型: {model}")
    print("输入 'resume <task_id>' 可恢复历史任务。")
    print("输入 'exit' 或 'quit' 退出。")
    print("=====================================================\n")

    current_memory = None
    current_task_desc = ""

    while True:
        try:
            user_input = input("\n[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[!] 收到退出信号，Agent 结束运行。")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("再见")
            break

        if user_input.startswith("resume "):
            task_id = user_input.split(" ", 1)[1].strip()
            try:
                current_memory = memory_manager.load_task(task_id)
                current_task_desc = current_memory.state.objective
                print(f"[*] 已成功挂载历史任务: {task_id}")
                user_input = "请继续当前任务"
            except Exception as e:
                print(f"[-] 恢复任务失败: {e}")
                continue

        elif not current_memory:
            current_task_desc = f"""
You are solving an authorized ONLINE web CTF challenge.
Target Info / User Request: {user_input}

Goals & Constraints:
1) Explore the website and find the REAL flag.
2) Write the candidate flag ONLY into: {candidate_flag_path}
3) Run pytest verification using: test_path = {test_path}, cwd = {test_cwd}
4) A candidate is valid ONLY if it matches regex: {flag_regex} and passes pytest.
""".strip()
            current_memory = memory_manager.create_task(current_task_desc)
            print(f"[*]创建新任务，Task ID: {current_memory.task_id}")

        else:
            current_memory.add_human_hint(user_input)
            print("[*]已记录入任务记忆。")

        last_messages = build_resume_messages(current_memory, user_input)

        agent = WebCTFAgent(
            client=client,
            tools=tools,
            runs_dir=os.getenv("RUNS_DIR", "./runs"),
            max_rounds=max_rounds,
            memory=current_memory,
            mcp_configs=mcp_servers,
        )

        print(f"[Agent]:思考与执行... ")
        try:
            result = await agent.solve(
                current_task_desc,
                resume_messages=last_messages if last_messages else None
            )

            print("\n" + "="*40)
            if result.get("ok"):
                print(f"🟢 [Agent 本阶段汇报]:\n{result.get('response', '')}")
            else:
                print(f"🔴 [Agent 执行暂停/异常]:\n{result.get('error')}")
            print("="*40)

            print(f"[*] 任务已暂停。当前任务 ID: {current_memory.task_id}")
            print(f"[*] 日志路径: {result.get('run_log')}")
            print("[*] 按回车让它继续。")

        except Exception as e:
            print(f"\n[!] 发生严重错误:\n{traceback.format_exc()}")
            print("[*] 状态已保存，您可以稍后通过 'resume' 恢复。")

if __name__ == "__main__":
    asyncio.run(main())