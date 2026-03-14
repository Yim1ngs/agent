from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from mcp import StdioServerParameters

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
            load_dotenv(dotenv_path=p, override=True)


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def main() -> None:
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

    mcp_servers = [
        # ==========================================
        # 1. Web 动态交互与渗透测试
        # ==========================================
        StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-puppeteer"],
        ),

        # ==========================================
        # 2. 极速纯文本/API 抓取
        # ==========================================
        #StdioServerParameters(
        #    command="npx",
        #    args=["-y", "@modelcontextprotocol/server-fetch"],
        #),

        # ==========================================
        # 3. 本地文件系统管理
        # 功能：赋予大模型在指定目录下自由新建、编辑、读取文件的能力。
        # ==========================================
        #StdioServerParameters(
        #    command="npx",
        #    args=["-y", "@modelcontextprotocol/server-filesystem", "./challenges"],
        #),

        # ==========================================
        # 4. GitHub 源码与 PoC 搜集
        # 功能：允许大模型直接在 GitHub 上搜索代码库、读取 Issues 和源码。
        # 提示：运行前最好在 .env 中配置 GITHUB_PERSONAL_ACCESS_TOKEN 提升 API 额度
        # ==========================================
        #StdioServerParameters(
        #    command="npx",
        #    args=["-y", "@modelcontextprotocol/server-github"],
        #),

        # ==========================================
        # 5. Brave Search 搜索引擎 (互联网冲浪与查资料)
        # 功能：赋予 Agent 直接搜索互联网的能力。
        # 提示：需要在 .env 中配置 BRAVE_API_KEY
        # ==========================================
        #StdioServerParameters(
        #    command="npx",
        #    args=["-y", "@modelcontextprotocol/server-brave-search"],
        #)

        # ==========================================
        # 6. SQLite 数据库分析
        # 功能：直接让大模型对本地的 .db 文件执行 SQL 语句。
        # ==========================================
        #StdioServerParameters(
        #    command="uvx",
        #    args=["mcp-server-sqlite", "--db-path", "./challenges/ctf_database.db"],
        #)
    ]

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
            f"1. The CURRENT TARGET URL is: {target_url}\n"
            f"   (If this URL is different from the history, you MUST use this new URL for all future requests!)\n"
            f"2. In the history above, past tool calls are shown as text `[Action: Tool ...]`. "
            f"This is ONLY a transcript. You CANNOT execute tools by typing `[Action: Tool ...]`. "
            f"You MUST strictly use the native JSON Tool Calling mechanism to make your next move!"
        )

        if human_hint:
            final_text = f"Human operator hint: {human_hint}" + system_warning
        else:
            final_text = "System Resume Notice: Continuing task." + system_warning

        if last_messages and last_messages[-1]["role"] == "user":
            if isinstance(last_messages[-1]["content"], list):
                last_messages[-1]["content"].append({"type": "text", "text": final_text})
            else:
                last_messages[-1]["content"] += f"\n\n{final_text}"
        else:
            last_messages.append({"role": "user", "content": final_text})

        agent = WebCTFAgent(
            client=client,
            tools=tools,
            runs_dir=os.getenv("RUNS_DIR", "./runs"),
            max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "20")),
            memory=memory,
            mcp_configs=mcp_servers,
        )

        result = await agent.solve(
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
            mcp_configs=mcp_servers,
        )

        result = await agent.solve(task_description)

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
    asyncio.run(main())