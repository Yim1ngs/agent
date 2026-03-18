from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import base64
import binascii
import hashlib
import urllib.parse
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


class _SimpleHTMLExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Dict[str, str]] = []
        self.forms: List[Dict[str, Any]] = []
        self._current_form: Optional[Dict[str, Any]] = None
        self.title_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_d = dict(attrs)

        if tag == "a":
            href = attrs_d.get("href", "")
            text_hint = attrs_d.get("title", "")
            self.links.append({"href": href, "text_hint": text_hint})

        elif tag == "form":
            self._current_form = {
                "action": attrs_d.get("action", ""),
                "method": (attrs_d.get("method") or "get").lower(),
                "inputs": [],
            }
            self.forms.append(self._current_form)

        elif tag == "input" and self._current_form is not None:
            self._current_form["inputs"].append(
                {
                    "name": attrs_d.get("name", ""),
                    "type": attrs_d.get("type", "text"),
                    "value": attrs_d.get("value", ""),
                }
            )

        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)


class WebToolRegistry:
    """
    Web-first CTF tool registry.

    Security model (recommended):
    - Read path must stay under workspace_root
    - Write path must stay under writable_roots (or default workspace-only rule)
    - HTTP host restriction is optional (None => unrestricted)
    """

    def __init__(
        self,
        workspace_root: str,
        allowed_hosts: Optional[List[str]] = None,
        writable_roots: Optional[List[str]] = None,
    ) -> None:
        self.workspace_root = os.path.abspath(workspace_root)

        if allowed_hosts:
            self.allowed_hosts: Optional[set[str]] = set(h.strip() for h in allowed_hosts if h.strip())
        else:
            self.allowed_hosts = None

        self.writable_roots = [os.path.abspath(p) for p in (writable_roots or [])]

        self.http = requests.Session()
        self.responses: Dict[str, Dict[str, Any]] = {}

    def _is_under(self, path: str, root: str) -> bool:
        try:
            common = os.path.commonpath([os.path.abspath(path), os.path.abspath(root)])
            return common == os.path.abspath(root)
        except ValueError:
            return False

    def _safe_path(self, path: str) -> str:
        full = os.path.abspath(path)
        if not self._is_under(full, self.workspace_root):
            raise ValueError(f"path out of workspace: {path}")
        return full

    def _safe_write_path(self, path: str) -> str:
        full = self._safe_path(path)

        if self.writable_roots:
            if not any(self._is_under(full, root) for root in self.writable_roots):
                raise ValueError(f"write path not allowed: {path}")
            return full

        rel = os.path.relpath(full, self.workspace_root)
        parts = [p.lower() for p in rel.split(os.sep)]
        if "workspace" not in parts:
            raise ValueError(
                f"write path not allowed (default policy only allows */workspace/*): {path}"
            )
        return full

    def _check_url_allowed(self, url: str) -> None:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            raise ValueError(f"unsupported scheme: {u.scheme}")

        if self.allowed_hosts is None:
            return

        host = u.hostname or ""
        if host not in self.allowed_hosts:
            raise ValueError(f"host not allowed: {host}. Allowed: {sorted(self.allowed_hosts)}")

    def _normalize_regex_patterns(self, regex_patterns: Any) -> List[str]:
        """
        Accept:
        - ["a", "b"]
        - "[\"a\", \"b\"]"   (model sometimes passes JSON-array as string)
        - "a"                (single regex)
        """
        if regex_patterns is None:
            return []

        if isinstance(regex_patterns, list):
            return [str(x) for x in regex_patterns]

        if isinstance(regex_patterns, str):
            s = regex_patterns.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return [str(x) for x in arr]
                except Exception:
                    pass
            return [s]

        return [str(regex_patterns)]

    def anthropic_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "http_request",
                "description": "Send an HTTP request. Stores response and returns response_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["GET", "POST"]},
                        "url": {"type": "string"},
                        "headers": {"type": "object"},
                        "params": {"type": "object"},
                        "data": {"type": "object"},
                        "json_body": {"type": "object"},
                        "timeout_sec": {"type": "integer"},
                        "allow_redirects": {"type": "boolean"},
                    },
                    "required": ["method", "url"],
                },
            },
            {
                "name": "extract_artifacts",
                "description": "Extract title/links/forms/regex hits from a stored HTTP response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response_id": {"type": "string"},
                        "regex_patterns": {
                            "oneOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "string"}
                            ]
                        },
                    },
                    "required": ["response_id"],
                },
            },
            {
                "name": "assert_http",
                "description": "Assert conditions on a stored HTTP response (status/body substrings/regex).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response_id": {"type": "string"},
                        "expected_status": {
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "array", "items": {"type": "integer"}},
                            ]
                        },
                        "must_contain": {"type": "array", "items": {"type": "string"}},
                        "must_not_contain": {"type": "array", "items": {"type": "string"}},
                        "regex_patterns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["response_id"],
                },
            },
            {
                "name": "get_response_body",
                "description": "Read a body slice from a stored response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response_id": {"type": "string"},
                        "start": {"type": "integer"},
                        "max_chars": {"type": "integer"},
                    },
                    "required": ["response_id"],
                },
            },
            {
                "name": "validate_candidate_flag",
                "description": "Validate a candidate flag string against a regex (default from env CTF_FLAG_REGEX).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "flag": {"type": "string"},
                        "regex": {"type": "string"},
                    },
                    "required": ["flag"],
                },
            },
            {
                "name": "write_file",
                "description": "Write text to a file. Only writable workspace paths are allowed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a text file under workspace_root.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "run_pytest",
                "description": "Run pytest verification. Relative test_path is resolved against cwd.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "test_path": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_sec": {"type": "integer"},
                    },
                    "required": ["test_path"],
                },
            },
            {
                "name": "text_encoder_decoder",
                "description": "Encode, decode, or hash a string. Extremely useful for CTF bypasses, token forging, or payload generation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "base64_encode", "base64_decode",
                                "url_encode", "url_decode",
                                "hex_encode", "hex_decode",
                                "md5", "sha1", "sha256"
                            ],
                            "description": "The operation to perform."
                        },
                        "text": {
                            "type": "string",
                            "description": "The input string to process."
                        }
                    },
                    "required": ["action", "text"],
                },
            },
            {
                "name": "execute_local_command",
                "description": "Run a local shell command or execute a script (e.g., python3 exp.py). Used for running generated exploit scripts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"},
                        "cwd": {"type": "string", "description": "Working directory (e.g., ./challenges)"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "wsl_shell",
                "description": (
                    "Execute a command inside WSL (Kali Linux). Use this for penetration testing tools "
                    "like nmap, sqlmap, gobuster, nikto, hydra, curl, etc. "
                    "Output is truncated to 2000 chars; redirect to a file and use grep/head for large outputs."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute inside WSL Kali (e.g., 'nmap -sV 192.168.1.1')"
                        },
                        "timeout_sec": {
                            "type": "integer",
                            "description": "Timeout in seconds (default 30, max 120)"
                        },
                        "distro": {
                            "type": "string",
                            "description": "WSL distribution name (default: kali-linux)"
                        }
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "python_sandbox",
                "description": (
                    "Execute arbitrary Python code in a local sandbox. The code is written to a temp .py file and run. "
                    "Use this for writing and running exploit scripts, data processing, crypto operations, etc. "
                    "Both stdout and stderr (including tracebacks) are captured and returned."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        },
                        "timeout_sec": {
                            "type": "integer",
                            "description": "Timeout in seconds (default 30, max 120)"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "search_knowledge",
                "description": (
                    "Search the local CTF knowledge base for techniques, payloads, and writeups. "
                    "Use this BEFORE blindly testing when you identify a vulnerability direction "
                    "(e.g., SQLi, SSTI, RCE, deserialization, prototype pollution). "
                    "Returns relevant knowledge articles matching keywords."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "Space-separated keywords to search (e.g., 'SSTI Jinja2 payload')"
                        },
                        "category": {
                            "type": "string",
                            "description": "Optional category filter (e.g., 'sqli', 'ssti', 'rce', 'deserialize', 'prototype_pollution', 'xss', 'reverse')"
                        }
                    },
                    "required": ["keywords"]
                }
            }
        ]

    def call(self, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fn = getattr(self, f"_tool_{name}")
        except AttributeError:
            return {"ok": False, "error": f"unknown tool: {name}"}

        try:
            return fn(**tool_input)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _tool_http_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout_sec: int = 10,
        allow_redirects: bool = True,
    ) -> Dict[str, Any]:
        self._check_url_allowed(url)
        method = str(method).upper()

        resp = self.http.request(
            method=method,
            url=url,
            headers=headers or {},
            params=params or {},
            data=data or None,
            json=json_body,
            timeout=timeout_sec,
            allow_redirects=allow_redirects,
        )

        rid = f"resp_{uuid.uuid4().hex[:12]}"
        body = resp.text
        record = {
            "response_id": rid,
            "status_code": resp.status_code,
            "url": str(resp.url),
            "headers": dict(resp.headers),
            "body": body,
        }
        self.responses[rid] = record

        return {
            "ok": True,
            "response_id": rid,
            "status_code": resp.status_code,
            "url": str(resp.url),
            "headers": dict(resp.headers),
            "body_preview": body[:1200],
            "body_len": len(body),
        }

    def _tool_extract_artifacts(
        self,
        response_id: str,
        regex_patterns: Optional[Any] = None,
    ) -> Dict[str, Any]:
        r = self.responses.get(response_id)
        if not r:
            return {"ok": False, "error": f"response_id not found: {response_id}"}

        body = r["body"]

        parser = _SimpleHTMLExtractor()
        try:
            parser.feed(body)
        except Exception:
            pass

        title = "".join(parser.title_parts).strip()
        links = parser.links[:50]
        forms = parser.forms[:20]

        patterns = self._normalize_regex_patterns(regex_patterns)
        regex_hits: Dict[str, List[str]] = {}

        for pat in patterns:
            try:
                hits = re.findall(pat, body, flags=re.IGNORECASE | re.DOTALL)
                regex_hits[pat] = [str(h) for h in hits[:20]]
            except re.error as e:
                regex_hits[pat] = [f"<regex_error: {e}>"]

        default_hits = {
            r"flag\{[A-Za-z0-9_\-]+\}": re.findall(r"flag\{[A-Za-z0-9_\-]+\}", body),
            r"Syc\{[A-Za-z0-9_\-]+\}": re.findall(r"Syc\{[A-Za-z0-9_\-]+\}", body),
            r"<!--.*?-->": re.findall(r"<!--.*?-->", body, flags=re.DOTALL),
            r"(csrf|token)[^\"\'\s<>:=]{0,20}": re.findall(
                r"(csrf|token)[^\"\'\s<>:=]{0,20}", body, flags=re.I
            ),
        }

        return {
            "ok": True,
            "response_id": response_id,
            "title": title,
            "links": links,
            "forms": forms,
            "regex_hits": regex_hits,
            "default_hits": default_hits,
        }

    def _tool_assert_http(
        self,
        response_id: str,
        expected_status: Optional[Any] = None,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
        regex_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r = self.responses.get(response_id)
        if not r:
            return {"ok": False, "error": f"response_id not found: {response_id}"}

        body = r["body"]
        status_code = r["status_code"]

        failures: List[str] = []
        regex_matches: Dict[str, List[str]] = {}

        if expected_status is not None:
            if isinstance(expected_status, int):
                allowed = [expected_status]
            else:
                allowed = [int(x) for x in expected_status]
            if status_code not in allowed:
                failures.append(f"status_code={status_code} not in {allowed}")

        for s in (must_contain or []):
            if s not in body:
                failures.append(f"body missing substring: {s!r}")

        for s in (must_not_contain or []):
            if s in body:
                failures.append(f"body contains forbidden substring: {s!r}")

        for pat in (regex_patterns or []):
            try:
                hits = re.findall(pat, body, flags=re.IGNORECASE | re.DOTALL)
                regex_matches[pat] = [str(x) for x in hits[:10]]
                if not hits:
                    failures.append(f"regex no match: {pat}")
            except re.error as e:
                failures.append(f"regex error {pat}: {e}")

        return {
            "ok": len(failures) == 0,
            "response_id": response_id,
            "status_code": status_code,
            "failures": failures,
            "regex_matches": regex_matches,
        }

    def _tool_get_response_body(
        self,
        response_id: str,
        start: int = 0,
        max_chars: int = 6000,
    ) -> Dict[str, Any]:
        r = self.responses.get(response_id)
        if not r:
            return {"ok": False, "error": f"response_id not found: {response_id}"}

        body = r["body"]
        start = max(0, int(start))
        max_chars = max(1, min(int(max_chars), 50000))
        chunk = body[start:start + max_chars]

        return {
            "ok": True,
            "response_id": response_id,
            "start": start,
            "returned_chars": len(chunk),
            "body_chunk": chunk,
            "total_len": len(body),
        }

    def _tool_validate_candidate_flag(self, flag: str, regex: Optional[str] = None) -> Dict[str, Any]:
        pat = regex or os.getenv("CTF_FLAG_REGEX", r"^flag\{[A-Za-z0-9_\-]+\}$")
        try:
            is_valid = re.fullmatch(pat, flag.strip()) is not None
        except re.error as e:
            return {"ok": False, "error": f"invalid regex: {e}"}

        return {
            "ok": True,
            "flag": flag,
            "regex": pat,
            "is_valid": is_valid,
        }

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self._safe_write_path(path)  # IMPORTANT: write restriction
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "ok": True,
            "path": p,
            "bytes": len(content.encode("utf-8")),
        }

    def _tool_read_file(self, path: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()

        return {
            "ok": True,
            "path": p,
            "content": txt[:20000],
        }

    def _tool_run_pytest(
        self,
        test_path: str,
        cwd: str = ".",
        timeout_sec: int = 30,
    ) -> Dict[str, Any]:
        """
        Path semantics:
        - cwd must be under workspace_root
        - if test_path is relative, resolve against cwd
        - resolved test file must be under workspace_root
        """
        cwd_abs = self._safe_path(cwd)

        if os.path.isabs(test_path):
            test_abs = self._safe_path(test_path)
        else:
            test_abs = os.path.abspath(os.path.join(cwd_abs, test_path))
            if not self._is_under(test_abs, self.workspace_root):
                raise ValueError(f"test_path out of workspace: {test_path}")

        proc = subprocess.run(
            ["pytest", "-q", test_abs],
            cwd=cwd_abs,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            shell=False,
        )

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    def _tool_text_encoder_decoder(self, action: str, text: str) -> Dict[str, Any]:
        """
        Versatile encoding, decoding, and hashing tool.
        """
        try:
            result = ""
            if action == "base64_encode":
                result = base64.b64encode(text.encode('utf-8')).decode('utf-8')
            elif action == "base64_decode":
                result = base64.b64decode(text.encode('utf-8')).decode('utf-8', errors='replace')
            elif action == "url_encode":
                result = urllib.parse.quote(text)
            elif action == "url_decode":
                result = urllib.parse.unquote(text)
            elif action == "hex_encode":
                result = binascii.hexlify(text.encode('utf-8')).decode('utf-8')
            elif action == "hex_decode":
                result = binascii.unhexlify(text.encode('utf-8')).decode('utf-8', errors='replace')
            elif action == "md5":
                result = hashlib.md5(text.encode('utf-8')).hexdigest()
            elif action == "sha1":
                result = hashlib.sha1(text.encode('utf-8')).hexdigest()
            elif action == "sha256":
                result = hashlib.sha256(text.encode('utf-8')).hexdigest()
            else:
                return {"ok": False, "error": f"Unknown action: {action}"}

            return {"ok": True, "action": action, "result": result}
        except Exception as e:
            return {"ok": False, "error": f"Encoding/Decoding failed: {e}"}

    def _tool_execute_local_command(self, command: str, cwd: str = "./challenges") -> Dict[str, Any]:
        """Runs a shell command safely inside the workspace."""
        try:
            safe_cwd = self._safe_path(cwd)

            proc = subprocess.run(
                command,
                cwd=safe_cwd,
                capture_output=True,
                text=True,
                timeout=30,
                shell=True
            )
            return {
                "ok": True,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-10000:],
                "stderr": proc.stderr[-10000:]
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command execution timed out after 30 seconds."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    MAX_OUTPUT_CHARS = 2000
    TRUNCATION_NOTICE = "\n\n[OUTPUT TRUNCATED at 2000 chars. Redirect to a file and use grep/head to view full output.]"

    def _tool_wsl_shell(
        self,
        command: str,
        timeout_sec: int = 30,
        distro: str = "kali-linux",
    ) -> Dict[str, Any]:
        """Execute a command inside WSL (Kali Linux) with timeout and output truncation."""
        timeout_sec = max(1, min(int(timeout_sec), 120))

        dangerous_patterns = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"]
        cmd_lower = command.lower()
        for pat in dangerous_patterns:
            if pat in cmd_lower:
                return {"ok": False, "error": f"Blocked dangerous command pattern: {pat}"}

        try:
            wsl_cmd = ["wsl", "-d", distro, "--", "bash", "-c", command]
            proc = subprocess.run(
                wsl_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                shell=False,
            )

            stdout = proc.stdout
            stderr = proc.stderr
            stdout_truncated = False
            stderr_truncated = False

            if len(stdout) > self.MAX_OUTPUT_CHARS:
                stdout = stdout[:self.MAX_OUTPUT_CHARS] + self.TRUNCATION_NOTICE
                stdout_truncated = True
            if len(stderr) > self.MAX_OUTPUT_CHARS:
                stderr = stderr[:self.MAX_OUTPUT_CHARS] + self.TRUNCATION_NOTICE
                stderr_truncated = True

            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"WSL command timed out after {timeout_sec}s. Consider increasing timeout or running in background with redirect.",
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "error": "WSL not found. Ensure WSL is installed and the distro is available.",
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _tool_python_sandbox(
        self,
        code: str,
        timeout_sec: int = 30,
    ) -> Dict[str, Any]:
        """Execute Python code in a temporary sandbox, capturing stdout and traceback."""
        timeout_sec = max(1, min(int(timeout_sec), 120))

        tmp_dir = os.path.join(self.workspace_root, "_sandbox_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        script_name = f"sandbox_{uuid.uuid4().hex[:8]}.py"
        script_path = os.path.join(tmp_dir, script_name)

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            proc = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=tmp_dir,
                shell=False,
            )

            stdout = proc.stdout
            stderr = proc.stderr

            if len(stdout) > self.MAX_OUTPUT_CHARS:
                stdout = stdout[:self.MAX_OUTPUT_CHARS] + self.TRUNCATION_NOTICE
            if len(stderr) > self.MAX_OUTPUT_CHARS:
                stderr = stderr[:self.MAX_OUTPUT_CHARS] + self.TRUNCATION_NOTICE

            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "script_path": script_path,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Python script timed out after {timeout_sec}s.",
                "script_path": script_path,
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            try:
                if os.path.exists(script_path):
                    os.remove(script_path)
            except OSError:
                pass

    def _tool_search_knowledge(
        self,
        keywords: str,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the local CTF knowledge base by keywords and optional category."""
        knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")

        if not os.path.isdir(knowledge_dir):
            return {"ok": False, "error": f"Knowledge base directory not found: {knowledge_dir}. Please create it."}

        search_words = set(keywords.lower().split())
        results = []

        search_dirs = []
        if category:
            cat_dir = os.path.join(knowledge_dir, category.lower().strip())
            if os.path.isdir(cat_dir):
                search_dirs.append(cat_dir)
            else:
                search_dirs.append(knowledge_dir)
        else:
            search_dirs.append(knowledge_dir)

        for search_dir in search_dirs:
            for root, _dirs, files in os.walk(search_dir):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue

                    content_lower = content.lower()
                    fname_lower = fname.lower()
                    score = 0
                    for word in search_words:
                        if word in fname_lower:
                            score += 3
                        score += content_lower.count(word)

                    if score > 0:
                        rel_path = os.path.relpath(fpath, knowledge_dir)
                        results.append({
                            "file": rel_path,
                            "score": score,
                            "content": content[:3000],
                        })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:5]

        if not results:
            return {
                "ok": True,
                "message": "No matching knowledge found. Try broader keywords or different category.",
                "available_categories": self._list_knowledge_categories(knowledge_dir),
            }

        return {
            "ok": True,
            "results": results,
            "total_matches": len(results),
        }

    @staticmethod
    def _list_knowledge_categories(knowledge_dir: str) -> List[str]:
        """List available knowledge categories."""
        if not os.path.isdir(knowledge_dir):
            return []
        return [
            d for d in os.listdir(knowledge_dir)
            if os.path.isdir(os.path.join(knowledge_dir, d)) and not d.startswith(".")
        ]