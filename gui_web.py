"""LocalCode Web GUI — pywebview + embedded HTML/CSS/JS."""

import json
import os
import threading

import webview

import config
import memory
import tools as tool_mod
from agent import Agent


class BackendAPI:
    def __init__(self):
        self.agent = Agent()
        self._processing = False
        self._lock = threading.Lock()
        self._perm_events = {}   # uid -> threading.Event
        self._perm_results = {}  # uid -> str
        import permission as _perm
        _perm.set_confirm_callback(self._perm_callback)

    def _perm_callback(self, tool_name: str, args: dict, risk_level: str, message: str, action_name: str) -> str:
        """Blocking callback — shows permission dialog in GUI, waits for user response."""
        import uuid as _uuid
        uid = str(_uuid.uuid4())[:8]
        event = threading.Event()
        self._perm_events[uid] = event
        # For bash commands, let LLM explain; explain goes in title, raw cmd in body
        explain = ""
        if tool_name == "run_bash":
            explain, message = self._explain_command(message)
        payload = json.dumps({
            "uid": uid,
            "tool_name": tool_name,
            "action_name": action_name,
            "risk_level": risk_level,
            "message": message,
            "explain": explain,
        }, ensure_ascii=False)
        code = f"showPermissionDialog({payload})"
        webview.windows[0].evaluate_js(code)
        event.wait()
        return self._perm_results.pop(uid, "deny")

    def _explain_command(self, command: str) -> tuple[str, str]:
        """Use intent model to translate shell command to Chinese.
        Returns (explanation, raw_command) or ('', command) on failure."""
        try:
            import config
            icfg = config.get_intent_config()
            if icfg:
                client = self.agent._make_client(icfg["provider"], icfg["model"], icfg["api_key"], icfg["base_url"])
            else:
                client = self.agent.client
            prompt = f"用中文翻译这条命令做了什么(操作代码多长，翻译就差不多长度）：\n{command[:300]}"
            msgs = [{"role": "user", "content": prompt}]
            text, _ = client.chat(msgs)
            if text and text.strip():
                return text.strip(), command
        except Exception:
            pass
        return "", command

    def permission_response(self, uid: str, result: str) -> str:
        """Called from JS when user responds to permission dialog."""
        self._perm_results[uid] = result
        if uid in self._perm_events:
            self._perm_events[uid].set()
        return "ok"

    def send_message(self, text: str, chat_history: str = "[]", images: str = "[]") -> str:
        """Returns JSON: {"messages": [...], "error": null}. images is JSON array of {media_type, data}."""
        if self._processing:
            return json.dumps({"messages": [], "error": "Still processing..."})

        with self._lock:
            self._processing = True
            try:
                import logger
                import agent
                import permission as _perm
                _perm.reset_round()
                imgs = []
                try:
                    imgs = json.loads(images)
                except Exception:
                    pass

                # Store images on agent — main agent decides when to call vision subagent
                if imgs:
                    self.agent.pending_images = imgs
                agent._current_agent = self.agent

                # Run intent subagent first — preprocess the query
                intent_result = self.agent.run_intent_subagent(text)
                main_text = text
                if intent_result:
                    main_text = f"[Intent analysis]\n{intent_result}\n\n[User message]\n{text}"
                    logger.info(f"intent analysis: {intent_result[:80]}...")

                # Select relevant memories for this query
                mem_ctx = self.agent.select_memories(text)
                if mem_ctx:
                    main_text = f"{mem_ctx}\n\n{main_text}"

                # User message is text-only, but hint about images if present
                if imgs:
                    main_text = f"[SYSTEM NOTE: The user attached {len(imgs)} image(s). You MUST call run_vision_subagent FIRST to see them before responding.]\n\n{main_text}"

                self.agent.messages.append({"role": "user", "content": main_text})
                logger.info(f"send_message: text={text[:60]}, images={len(imgs)}")
                msgs = []
                try:
                    prev = json.loads(chat_history)
                except Exception:
                    prev = []

                for _ in range(15):
                    txt, tool_calls = self.agent.client.chat(self.agent.messages)

                    if tool_calls:
                        anthropic_content = []
                        if txt:
                            anthropic_content.append({"type": "text", "text": txt})
                        for tc in tool_calls:
                            anthropic_content.append({
                                "type": "tool_use", "id": tc["id"],
                                "name": tc["name"], "input": tc["arguments"],
                            })
                        self.agent.messages.append({
                            "role": "assistant", "content": anthropic_content
                        })

                        for tc in tool_calls:
                            result = tool_mod.registry.execute(tc["name"], tc["arguments"])
                            msgs.append({
                                "type": "tool",
                                "name": tc["name"],
                                "args": json.dumps(tc["arguments"], ensure_ascii=False),
                                "result": result[:600],
                            })
                            self.agent.messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": result,
                                }],
                            })
                        # Continue loop — don't return early, wait for final text-only response
                    else:
                        self.agent.messages.append({"role": "assistant", "content": txt})
                        msgs.append({"type": "assistant", "content": txt})
                        self._processing = False
                        self._save_now(prev + msgs)
                        # Auto-extract memories from dialogue
                        self.agent.extract_memories_after_turn()
                        return json.dumps({"messages": msgs, "error": None}, ensure_ascii=False)

                self._processing = False
                return json.dumps({"messages": msgs, "error": "Max tool rounds reached."}, ensure_ascii=False)
            except Exception as e:
                self._processing = False
                return json.dumps({"messages": [], "error": str(e)}, ensure_ascii=False)

    def get_config(self) -> str:
        cfg = config.load_config()
        safe = {
            "provider": cfg.get("provider", ""),
            "model": cfg.get("model", ""),
            "openai_base_url": cfg.get("openai_base_url", ""),
            "has_key": bool(config.get_api_key(cfg.get("provider", ""))),
            "intent_provider": cfg.get("intent_provider", ""),
            "intent_model": cfg.get("intent_model", ""),
            "intent_base_url": cfg.get("intent_base_url", ""),
            "vision_provider": cfg.get("vision_provider", ""),
            "vision_model": cfg.get("vision_model", ""),
            "vision_base_url": cfg.get("vision_base_url", ""),
            "compress_keep": cfg.get("compress_keep", 10),
            "conversation_path": cfg.get("conversation_path", ""),
        }
        return json.dumps(safe, ensure_ascii=False)

    def save_settings(self, provider: str, model: str, api_key: str, base_url: str,
                      intent_provider: str = "", intent_model: str = "", intent_key: str = "", intent_url: str = "",
                      vision_provider: str = "", vision_model: str = "", vision_key: str = "", vision_url: str = "",
                      compress_keep: str = "10", conversation_path: str = "") -> str:
        cfg = config.load_config()
        if provider == "anthropic":
            cfg["provider"] = "anthropic"
            if api_key: cfg["anthropic_api_key"] = api_key
            cfg["model"] = model
        else:
            cfg["provider"] = "openai"
            if api_key: cfg["openai_api_key"] = api_key
            cfg["model"] = model
            cfg["openai_base_url"] = base_url

        # Intent API
        cfg["intent_provider"] = intent_provider
        cfg["intent_model"] = intent_model
        if intent_key: cfg["intent_api_key"] = intent_key
        cfg["intent_base_url"] = intent_url

        # Vision API
        cfg["vision_provider"] = vision_provider
        cfg["vision_model"] = vision_model
        if vision_key: cfg["vision_api_key"] = vision_key
        cfg["vision_base_url"] = vision_url

        # Compress
        try: cfg["compress_keep"] = int(compress_keep)
        except: pass

        # Storage
        cfg["conversation_path"] = conversation_path

        config.save_config(cfg)
        try:
            self.agent.client._setup_client()
        except Exception as e:
            return f"Error reloading client: {e}"
        return "ok"

    def get_memories(self) -> str:
        entries = memory.list_memories()
        return json.dumps(entries, ensure_ascii=False)

    def _conversation_file(self) -> Path:
        from pathlib import Path
        cfg = config.load_config()
        custom = cfg.get("conversation_path", "")
        if custom:
            p = Path(custom)
            if p.is_dir():
                p = p / "conversation.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        d = Path.home() / ".localcode"
        d.mkdir(parents=True, exist_ok=True)
        return d / "conversation.json"

    def save_conversation(self, chat_msgs: str) -> str:
        """Save both display messages and agent context."""
        from pathlib import Path
        try:
            data = {
                "chat": json.loads(chat_msgs),
                "agent_messages": self.agent.messages,
            }
            f = self._conversation_file()
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return "ok"
        except Exception as e:
            return str(e)

    def load_conversation(self) -> str:
        """Return saved conversation as JSON. {chat: [...], agent_messages: [...]}"""
        f = self._conversation_file()
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                # Restore agent context
                if "agent_messages" in data:
                    self.agent.messages = data["agent_messages"]
                return json.dumps(data.get("chat", []), ensure_ascii=False)
            except Exception:
                pass
        return "[]"

    def _save_now(self, chat_msgs: list):
        """Save conversation in background (excludes tool messages)."""
        try:
            f = self._conversation_file()
            clean = [m for m in chat_msgs if m.get("type") != "tool"]
            data = {
                "chat": clean,
                "agent_messages": self.agent.messages,
            }
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def compress_context(self, keep: int = 10) -> str:
        """Keep last `keep` messages, compress older ones into a summary.
        Returns JSON: {"summary": "...", "kept": N, "compressed": N}"""
        msgs = self.agent.messages
        # msgs[0] = system prompt, then user/assistant messages
        if len(msgs) <= keep + 1:
            return json.dumps({"summary": "", "kept": len(msgs) - 1, "compressed": 0}, ensure_ascii=False)

        old = msgs[1:-keep]  # exclude system prompt and last N
        recent = msgs[-keep:]

        # Build a summary request
        old_text = []
        for m in old:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        old_text.append(f"[{role}]: {b['text'][:500]}")
                    elif isinstance(b, dict) and b.get("type") == "tool_use":
                        old_text.append(f"[{role} tool]: {b.get('name','?')}")
            else:
                old_text.append(f"[{role}]: {str(content)[:500]}")

        summary_prompt = (
            "Summarize this conversation history concisely, preserving key decisions, "
            "code patterns, user preferences, and important context:\n\n" +
            "\n".join(old_text)
        )

        try:
            # Use a quick single-turn to get summary
            summary_msgs = [
                {"role": "user", "content": summary_prompt},
            ]
            txt, _ = self.agent.client.chat(summary_msgs)
            summary = txt or "(summary unavailable)"
        except Exception as e:
            summary = f"(compression failed: {e})"

        # Rebuild: system prompt + summary + recent
        system = msgs[0]
        self.agent.messages = [
            system,
            {"role": "user", "content": f"[Earlier conversation summary]\n{summary}"},
            {"role": "assistant", "content": "Understood. Continuing with the context above."},
            *recent,
        ]

        return json.dumps({
            "summary": summary[:300],
            "kept": len(recent),
            "compressed": len(old),
        }, ensure_ascii=False)

    def clear_context(self) -> str:
        self.agent.clear()
        f = self._conversation_file()
        if f.exists():
            f.unlink()
        return "ok"

    def get_tool_list(self) -> str:
        return ", ".join(tool_mod.registry.list_names())

    def copy_to_clipboard(self, text: str) -> str:
        import subprocess
        # Use Windows clip.exe as reliable fallback
        try:
            p = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
            p.communicate(input=text.encode('utf-16-le', errors='replace'))
            return "ok"
        except Exception:
            # try tkinter
            try:
                import tkinter as tk
                r = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(text)
                r.update()
                r.destroy()
                return "ok"
            except Exception:
                return "failed"

    def get_context(self) -> str:
        """Return current agent messages for debugging."""
        msgs = []
        for m in self.agent.messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        t = b.get("type", "?")
                        if t == "text":
                            parts.append(b.get("text", "")[:200])
                        elif t == "tool_use":
                            parts.append(f"[tool: {b.get('name','?')}]")
                        elif t == "tool_result":
                            parts.append(f"[result: {b.get('content','')[:100]}]")
                        elif t == "image":
                            parts.append("[image]")
                    else:
                        parts.append(str(b)[:100])
                content = " | ".join(parts)
            msgs.append({"role": role, "content": str(content)[:500]})
        return json.dumps(msgs, ensure_ascii=False)

    def get_logs(self) -> str:
        import logger
        return logger.tail(50)

    # ── Skills ─────────────────────────────────────────────────

    def list_skills(self) -> str:
        import skills
        return json.dumps(skills.list_skills(), ensure_ascii=False)

    def save_skill(self, name: str, description: str, prompt: str) -> str:
        import skills
        return skills.save_skill(name, description, prompt)

    # ── MCP ────────────────────────────────────────────────────

    def list_mcp_servers(self) -> str:
        import mcp_client
        configs = mcp_client.load_servers()
        connected = mcp_client.list_connected()
        conn_names = {c["name"] for c in connected}
        result = []
        for s in configs:
            result.append({
                "name": s["name"],
                "description": s.get("description", ""),
                "command": s.get("command", ""),
                "status": "connected" if s["name"] in conn_names else "disconnected",
            })
        return json.dumps(result, ensure_ascii=False)

    def add_mcp_server(self, name: str, command: str, args: str = "[]",
                       env: str = "{}", description: str = "") -> str:
        import json as _json
        import mcp_client
        try:
            a = _json.loads(args)
        except Exception:
            a = []
        try:
            e = _json.loads(env)
        except Exception:
            e = {}
        return mcp_client.add_server(name, command, a, e, description)

    def connect_mcp_server(self, name: str = "") -> str:
        import mcp_client
        if name:
            servers = mcp_client.load_servers()
            for s in servers:
                if s["name"] == name:
                    return mcp_client.connect_server(s)
            return f"Server '{name}' not found."
        return mcp_client.auto_connect_all()


# ── HTML (embedded for exe compatibility) ─────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
:root{--bg:#1e1e1e;--bg2:#252526;--bg3:#2d2d2d;--fg:#ccc;--fg2:#e0e0e0;--border:#3e3e3e;--accent:#007acc;--ah:#1a8ad4;--blue:#569cd6;--green:#4ec9b0;--orange:#ce9178;--purple:#c586c0;--gray:#808080;--red:#f44747;--yellow:#dcdcaa}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;overflow:hidden}
.titlebar{background:var(--bg2);height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid var(--border);user-select:none}
.titlebar span{font-size:13px;font-weight:600;color:var(--fg2)}
.titlebar div{display:flex;gap:6px}
.titlebar button{background:var(--bg3);border:1px solid var(--border);color:var(--fg);padding:4px 10px;font-size:12px;border-radius:4px;cursor:pointer}
.titlebar button:hover{background:#4a4a4a}
.chat{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:8px}
.chat::-webkit-scrollbar{width:6px}.chat::-webkit-scrollbar-track{background:transparent}.chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.msg{display:flex;gap:8px;animation:fadeIn .2s ease;max-width:90%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.msg.assistant,.msg.tool,.msg.tool-result{align-self:flex-start}
.msg.tool-result{margin-left:24px}
.avatar{width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:bold;flex-shrink:0}
.avatar.user{background:var(--blue);color:#fff}
.avatar.assistant{background:var(--green);color:#1e1e1e}
.avatar.tool{background:var(--orange);color:#1e1e1e}
.avatar.tool-result{background:transparent;color:var(--gray);font-size:10px}
.bubble{padding:8px 14px;border-radius:12px;line-height:1.5;font-size:13px;white-space:pre-wrap;word-break:break-word}
.bubble.user{background:var(--accent);color:#fff;border-bottom-right-radius:4px}
.bubble.assistant{background:var(--bg2);color:var(--fg);border-bottom-left-radius:4px}
.bubble.tool{background:#2a1f0a;color:var(--orange);border:1px solid #3d2e14;font-family:'Cascadia Code','Consolas',monospace;font-size:12px}
.bubble.tool-result{background:var(--bg3);color:var(--gray);font-size:12px;max-height:120px;overflow-y:auto}
.tool-collapsed{display:none !important}
.detail-toggle{font-size:11px;color:var(--gray);cursor:pointer;margin-top:4px;user-select:none}
.detail-toggle:hover{color:var(--orange)}
.bubble pre{background:rgba(0,0,0,.3);padding:10px;border-radius:6px;overflow-x:auto;margin:4px 0}
.input-area{background:var(--bg2);border-top:1px solid var(--border);padding:12px 16px;display:flex;gap:8px;align-items:flex-end}
.input-area textarea{flex:1;background:var(--bg3);color:var(--fg);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-family:inherit;font-size:13px;resize:none;min-height:40px;max-height:150px;outline:none;line-height:1.4}
.input-area textarea:focus{border-color:var(--accent)}
.input-area button{background:var(--accent);color:#fff;border:none;padding:10px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap}
.input-area button:hover{background:var(--ah)}
.input-area button:disabled{opacity:.5;cursor:not-allowed}
.statusbar{background:var(--bg2);border-top:1px solid var(--border);padding:3px 12px;font-size:11px;color:var(--gray);display:flex;gap:16px}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center}
.modal-overlay.active{display:flex}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:24px;width:440px;max-height:80vh;overflow-y:auto;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.modal h2{color:var(--fg2);margin-bottom:16px;font-size:16px}
.modal label{display:block;color:var(--fg2);font-size:12px;font-weight:600;margin:12px 0 4px;text-transform:uppercase;letter-spacing:.5px}
.modal select,.modal input{width:100%;background:var(--bg3);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-size:13px;outline:none}
.settings-layout{display:flex;gap:0;min-height:380px}
.settings-sidebar{width:140px;flex-shrink:0;border-right:1px solid var(--border);padding-right:0}
.settings-sidebar .cat-btn{display:block;width:100%;text-align:left;padding:10px 14px;background:none;border:none;color:var(--gray);font-size:13px;cursor:pointer;border-radius:6px 0 0 6px;margin-bottom:2px}
.settings-sidebar .cat-btn:hover{background:var(--bg3);color:var(--fg2)}
.settings-sidebar .cat-btn.active{background:var(--bg3);color:var(--fg);font-weight:600}
.settings-content{flex:1;padding:0 0 0 20px;overflow-y:auto;max-height:60vh}
.settings-tab{display:none}
.settings-tab.active{display:block}
.modal select:focus,.modal input:focus{border-color:var(--accent)}
.modal .btn-row{display:flex;gap:8px;justify-content:flex-end;margin-top:20px}
.modal button{padding:8px 20px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;border:none}
.modal .btn-cancel{background:var(--bg3);color:var(--fg)}
.modal .btn-save{background:var(--accent);color:#fff}.modal .btn-save:hover{background:var(--ah)}
.memory-list{list-style:none}
.memory-list li{padding:8px 10px;border-bottom:1px solid var(--border);font-size:13px}
.memory-list li:hover{background:var(--bg3)}
.typing{display:none;align-self:flex-start;margin-left:36px;flex-direction:column;gap:2px}
.typing.active{display:flex}
.typing .dots{display:flex;gap:4px;padding:4px 14px}
.typing .dots span{width:6px;height:6px;background:var(--gray);border-radius:50%;animation:bounce 1.4s infinite ease-in-out}
.typing .dots span:nth-child(2){animation-delay:.2s}.typing .dots span:nth-child(3){animation-delay:.4s}
.typing .status{font-size:10px;color:var(--gray);padding:0 14px}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@keyframes bounce{0%,80%,100%{transform:scale(.6)}40%{transform:scale(1)}}
.msg-actions{display:flex;gap:2px;margin-top:2px;opacity:0;transition:opacity .15s}
.msg:hover .msg-actions{opacity:1}
.msg-actions button{background:transparent;border:none;color:var(--gray);font-size:10px;padding:2px 6px;cursor:pointer;border-radius:3px}
.msg-actions button:hover{background:var(--bg3);color:var(--fg)}
.msg-actions .quote-cb{width:12px;height:12px;margin:2px 4px 0 0;accent-color:var(--accent);cursor:pointer}
.empty-state{flex:1;display:flex;align-items:center;justify-content:center;color:var(--gray);font-size:14px;text-align:center;line-height:1.8}
.empty-state strong{color:var(--fg2)}
.img-preview{display:none;gap:6px;padding:6px 16px 0;flex-wrap:wrap}
.img-preview.has-images{display:flex}
.img-thumb{position:relative;width:60px;height:60px;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.img-thumb img{width:100%;height:100%;object-fit:cover}
.img-thumb .rm{position:absolute;top:0;right:0;background:rgba(0,0,0,.7);color:#fff;border:none;width:16px;height:16px;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;border-radius:0 0 0 4px}
.msg-img{max-width:200px;max-height:160px;border-radius:8px;margin:4px 0;cursor:pointer}
.input-area .attach-btn{background:var(--bg3);color:var(--fg);border:1px solid var(--border);padding:10px 12px;border-radius:8px;font-size:14px;cursor:pointer;white-space:nowrap}
.input-area .attach-btn:hover{background:#4a4a4a}
.context-modal .ctx-section{margin:12px 0}
.ctx-section h3{color:var(--fg2);font-size:13px;margin-bottom:4px}
.ctx-section .ctx-item{font-size:12px;color:var(--gray);padding:2px 0;white-space:pre-wrap;word-break:break-all;max-height:80px;overflow-y:auto;border-left:2px solid var(--border);padding-left:8px}
.perm-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:200;align-items:center;justify-content:center}
.perm-overlay.active{display:flex}
.perm-dialog{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px 24px;width:480px;box-shadow:0 8px 32px rgba(0,0,0,.6)}
.perm-dialog .perm-icon{font-size:28px;margin-bottom:8px}
.perm-dialog .perm-title{color:var(--fg2);font-size:14px;font-weight:600;margin-bottom:2px;white-space:pre-wrap;line-height:1.5}
.perm-dialog .perm-tool{color:var(--orange);font-size:12px;font-family:'Cascadia Code','Consolas',monospace;margin-bottom:4px}
.perm-dialog .perm-section-label{color:var(--gray);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:8px;margin-bottom:4px}
.perm-dialog .perm-msg{color:var(--fg);font-size:13px;margin-bottom:8px;padding:10px 12px;background:var(--bg3);border-radius:6px;border-left:3px solid var(--yellow);max-height:160px;overflow-y:auto;white-space:pre-wrap;line-height:1.6}
.perm-dialog .perm-options{display:flex;flex-direction:column;gap:4px;margin:12px 0}
.perm-dialog .perm-opt{padding:10px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:6px;color:var(--fg);font-size:13px;cursor:pointer;display:flex;align-items:center;gap:8px;transition:all .1s}
.perm-dialog .perm-opt:hover{background:#383838}
.perm-dialog .perm-opt.selected{background:var(--accent);border-color:var(--ah);color:#fff}
.perm-dialog .perm-opt .opt-key{font-size:10px;color:var(--gray);background:var(--bg2);border-radius:4px;padding:2px 6px;min-width:20px;text-align:center}
.perm-dialog .perm-opt.selected .opt-key{color:#fff;background:rgba(255,255,255,.2)}
.perm-dialog .perm-opt .opt-label{flex:1}
.perm-dialog .perm-opt .opt-hint{font-size:10px;color:var(--gray)}
.perm-dialog .perm-opt.selected .opt-hint{color:rgba(255,255,255,.7)}
.perm-dialog .perm-footer{color:var(--gray);font-size:10px;margin-top:6px;text-align:center}
.perm-dialog .perm-footer kbd{background:var(--bg3);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-family:inherit;font-size:10px}
</style>
</head>
<body>

<div class="titlebar">
  <span>LocalCode</span>
  <div>
    <button onclick="openSkills()">Skills</button>
    <button onclick="openLogs()">Logs</button>
    <button onclick="openMemories()">Memories</button>
    <button onclick="compress()">Compress</button>
    <button onclick="openSettings()">Settings</button>
    <button onclick="toggleContext()">上下文</button>
    <button onclick="clearChat()">Clear</button>
  </div>
</div>

<div class="chat" id="chat">
  <div class="empty-state" id="empty">
    <div>
      <strong>LocalCode</strong><br>
      Ask me to write, read, or edit code.<br>
      <span style="font-size:12px;color:var(--gray)">Ctrl+Enter to send</span>
    </div>
  </div>
  <div class="typing" id="typing"><div class="dots"><span></span><span></span><span></span></div><div class="status" id="typing-status"></div></div>
</div>

<div class="img-preview" id="img-preview"></div>
<div class="input-area">
  <button class="attach-btn" onclick="document.getElementById('file-input').click()" title="Attach images">+Img</button>
  <input type="file" id="file-input" accept="image/*" multiple style="display:none" onchange="handleFileSelect(event)">
  <textarea id="input" placeholder="Type a message... (Ctrl+Enter to send, Ctrl+V paste image)" rows="1" oninput="this.style.height='auto';this.style.height=Math.min(this.scrollHeight,150)+'px'"></textarea>
  <button id="send-btn" onclick="sendMsg()">Send</button>
</div>

<div class="statusbar">
  <span id="st-main">Main: ...</span>
  <span id="st-intent" style="color:var(--orange)"></span>
  <span id="st-vision" style="color:var(--yellow)"></span>
</div>

<div class="modal-overlay" id="settings-modal">
  <div class="modal" style="width:680px;padding:16px 20px">
    <h2 style="margin-bottom:12px">Settings</h2>
    <div class="settings-layout">
      <div class="settings-sidebar">
        <button class="cat-btn active" data-tab="main" onclick="switchSettingsTab('main')">Main API</button>
        <button class="cat-btn" data-tab="intent" onclick="switchSettingsTab('intent')">Intent API</button>
        <button class="cat-btn" data-tab="vision" onclick="switchSettingsTab('vision')">Vision API</button>
        <button class="cat-btn" data-tab="storage" onclick="switchSettingsTab('storage')">Storage</button>
        <button class="cat-btn" data-tab="compress" onclick="switchSettingsTab('compress')">Compress</button>
      </div>
      <div class="settings-content">
        <div class="settings-tab active" id="tab-main">
          <label>Provider</label>
          <select id="s-prov" onchange="onProvChange()">
            <option value="anthropic">Anthropic (Claude)</option>
            <option value="openai">OpenAI (GPT)</option>
            <option value="deepseek">DeepSeek</option>
            <option value="qwen">Qwen</option>
            <option value="custom">Other</option>
          </select>
          <label>Model</label><input id="s-model" placeholder="e.g. gpt-4o">
          <label>API Key</label><input id="s-key" type="password" placeholder="sk-...">
          <label>Base URL</label><input id="s-url" placeholder="https://api.openai.com/v1">
        </div>
        <div class="settings-tab" id="tab-intent">
          <label>Provider</label>
          <select id="s-iprov" onchange="onIProvChange()">
            <option value="">Same as main</option>
            <option value="anthropic">Anthropic (Claude)</option>
            <option value="openai">OpenAI (GPT)</option>
            <option value="deepseek">DeepSeek</option>
            <option value="qwen">Qwen</option>
            <option value="custom">Other</option>
          </select>
          <label>Model</label><input id="s-imodel" placeholder="e.g. gpt-4o-mini">
          <label>API Key</label><input id="s-ikey" type="password" placeholder="(empty = use main)">
          <label>Base URL</label><input id="s-iurl" placeholder="https://api.openai.com/v1">
        </div>
        <div class="settings-tab" id="tab-vision">
          <label>Provider</label>
          <select id="s-vprov" onchange="onVProvChange()">
            <option value="">Same as main</option>
            <option value="anthropic">Anthropic (Claude)</option>
            <option value="openai">OpenAI (GPT-4V)</option>
            <option value="qwen">Qwen-VL</option>
            <option value="custom">Other</option>
          </select>
          <label>Model</label><input id="s-vmodel" placeholder="e.g. gpt-4o">
          <label>API Key</label><input id="s-vkey" type="password" placeholder="(empty = use main)">
          <label>Base URL</label><input id="s-vurl" placeholder="https://api.openai.com/v1">
        </div>
        <div class="settings-tab" id="tab-storage">
          <label>Conversation save path</label>
          <input id="s-convpath" placeholder="D:\conversations (directory or .json file)">
          <span style="font-size:11px;color:var(--gray)">目录或文件路径均可。留空用默认 (~/.localcode/conversation.json)</span>
        </div>
        <div class="settings-tab" id="tab-compress">
          <label>Recent messages to keep</label>
          <input id="s-keep" type="number" min="2" max="50" placeholder="10" style="width:80px">
          <span style="font-size:11px;color:var(--gray);margin-left:8px">Older messages summarized in LLM context</span>
        </div>
      </div>
    </div>
    <div class="btn-row" style="margin-top:16px">
      <button class="btn-cancel" onclick="closeSettings()">Cancel</button>
      <button class="btn-save" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="logs-modal">
  <div class="modal" style="width:600px">
    <h2>Backend Logs</h2>
    <pre id="log-content" style="background:var(--bg);color:var(--fg);padding:12px;border-radius:6px;font-size:11px;font-family:'Cascadia Code','Consolas',monospace;max-height:400px;overflow-y:auto;white-space:pre-wrap">Loading...</pre>
    <div class="btn-row">
      <button class="btn-cancel" onclick="refreshLogs()">Refresh</button>
      <button class="btn-cancel" onclick="closeLogs()">Close</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="skills-modal">
  <div class="modal">
    <h2>Skill Library</h2>
    <ul class="memory-list" id="skill-list"></ul>
    <div class="btn-row" style="margin-top:12px">
      <button class="btn-cancel" onclick="addNewSkill()">+ New Skill</button>
      <button class="btn-cancel" onclick="closeSkills()">Close</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="mcp-modal">
  <div class="modal" style="width:520px">
    <h2>MCP Servers</h2>
    <ul class="memory-list" id="mcp-list"></ul>
    <div style="margin-top:12px">
      <label>Add Server</label>
      <input id="mcp-name" placeholder="Server name">
      <input id="mcp-cmd" placeholder="Command (e.g. npx)" style="margin-top:6px">
      <input id="mcp-args" placeholder="Args JSON (e.g. [\"-y\", \"@anthropic-ai/mcp-server-fetch\"])" style="margin-top:6px">
    </div>
    <div class="btn-row" style="margin-top:12px">
      <button class="btn-cancel" onclick="addMcpServer()">Add Server</button>
      <button class="btn-cancel" onclick="connectMcpAll()">Connect All</button>
      <button class="btn-cancel" onclick="closeMcp()">Close</button>
    </div>
  </div>
</div>

<div class="modal-overlay" id="context-modal">
  <div class="modal context-modal">
    <h2>当前上下文</h2>
    <div id="ctx-content"></div>
    <div class="btn-row"><button class="btn-cancel" onclick="closeContext()">关闭</button></div>
  </div>
</div>

<div class="modal-overlay" id="memories-modal">
  <div class="modal">
    <h2>Persistent Memories</h2>
    <ul class="memory-list" id="memory-list"></ul>
    <div class="btn-row"><button class="btn-cancel" onclick="closeMemories()">Close</button></div>
  </div>
</div>

<div class="perm-overlay" id="perm-overlay">
  <div class="perm-dialog" id="perm-dialog">
    <div class="perm-icon" id="perm-icon"></div>
    <div class="perm-title" id="perm-title"></div>
    <div class="perm-section-label" id="perm-section-label">此操作将：</div>
    <div class="perm-msg" id="perm-msg"></div>
    <div class="perm-options" id="perm-options"></div>
    <div class="perm-footer">
      <kbd>&uarr;</kbd><kbd>&darr;</kbd> 选择 &emsp; <kbd>Shift</kbd>+<kbd>Enter</kbd> 确认 &emsp; <kbd>Esc</kbd> 拒绝
    </div>
  </div>
</div>

<script>
const PRESETS={anthropic:{p:'anthropic',m:'claude-sonnet-4-6',u:''},openai:{p:'openai',m:'gpt-4o',u:'https://api.openai.com/v1'},deepseek:{p:'openai',m:'deepseek-chat',u:'https://api.deepseek.com/v1'},qwen:{p:'openai',m:'qwen-plus',u:'https://dashscope.aliyuncs.com/compatible-mode/v1'},custom:{p:'openai',m:'gpt-4o',u:'https://api.openai.com/v1'}};
const VPRESETS={anthropic:{p:'anthropic',m:'claude-sonnet-4-6',u:''},openai:{p:'openai',m:'gpt-4o',u:'https://api.openai.com/v1'},qwen:{p:'openai',m:'qwen-vl-plus',u:'https://dashscope.aliyuncs.com/compatible-mode/v1'},custom:{p:'openai',m:'gpt-4o',u:'https://api.openai.com/v1'}};
const IPRESETS={anthropic:{p:'anthropic',m:'claude-sonnet-4-6',u:''},openai:{p:'openai',m:'gpt-4o-mini',u:'https://api.openai.com/v1'},deepseek:{p:'openai',m:'deepseek-chat',u:'https://api.deepseek.com/v1'},qwen:{p:'openai',m:'qwen-plus',u:'https://dashscope.aliyuncs.com/compatible-mode/v1'},custom:{p:'openai',m:'gpt-4o-mini',u:'https://api.openai.com/v1'}};
var _ready=false,_chatHistory=[];
window.addEventListener('pywebviewready',function(){_ready=true;init();});
setTimeout(function(){if(!_ready){_ready=true;init();}},3000);

async function init(){
  try{
    var r=await window.pywebview.api.get_config();
    var c=JSON.parse(r);
    document.getElementById('st-main').textContent='Main: '+(c.model||'?')+' @ '+(c.provider||'?');
    var im=c.intent_model;document.getElementById('st-intent').textContent=im?'| Intent: '+im+' @ '+(c.intent_provider||'?'):'';
    var vm=c.vision_model;document.getElementById('st-vision').textContent=vm?'| Vision: '+vm+' @ '+(c.vision_provider||'?'):'';
    if(!c.has_key)document.getElementById('empty').innerHTML='<div><strong>Welcome!</strong><br>Click <b>Settings</b> to configure your API key.</div>';
  }catch(e){document.getElementById('st-main').textContent='Disconnected';}
  // Load saved conversation
  try{
    var h=await window.pywebview.api.load_conversation();
    var msgs=JSON.parse(h);
    if(msgs.length>0){
      document.getElementById('empty').style.display='none';
      for(var i=0;i<msgs.length;i++){
        var m=msgs[i];
        if(m.type==='tool')continue;
        else addMsg(m.role||'assistant',m.content,m.images);
      }
      _chatHistory=msgs;
    }
  }catch(e){}
}

async function sendMsg(){
  var t=document.getElementById('input');
  var v=t.value.trim();
  if(!v&&_images.length===0)return;
  if(v==='/clear'){t.value='';clearChat();return;}
  if(v==='/settings'||v==='/config'){t.value='';openSettings();return;}
  if(v==='/memories'||v==='/memory'){t.value='';openMemories();return;}
if(v==='/compress'||v.startsWith('/compress ')){t.value='';var n=parseInt(v.split(' ')[1])||10;compress(n);return;}
if(v==='/logs'){t.value='';openLogs();return;}
  // Strip quote summary prefix, build full message with quoted content
  var realMsg=v.replace(/^\[已选中\d+条消息\][\s\S]*?\n/,'').trim()||v;
  var fullMsg=realMsg;
  if(_quoted.length>0){
    var quotedTexts=[];
    for(var i=0;i<_quoted.length;i++)quotedTexts.push('[引用'+(i+1)+']\n'+_quoted[i].text);
    fullMsg=realMsg+'\n\n--- 引用的对话 ---\n'+quotedTexts.join('\n\n');
  }
  var currentImages=_images.slice();_images=[];renderImagePreview();
  // Merge quoted images too
  for(var i=0;i<_quoted.length;i++){
    var qims=_quoted[i].images||[];
    for(var j=0;j<qims.length;j++)currentImages.push(qims[j]);
  }
  addMsg('user',realMsg,currentImages);t.value='';t.style.height='auto';
  _chatHistory.push({role:'user',content:realMsg,images:currentImages.length>0?currentImages:undefined});
  document.getElementById('empty').style.display='none';
  setTyping(true,'Intent analysis + Main agent processing...');document.getElementById('send-btn').disabled=true;
  try{
    var r=await window.pywebview.api.send_message(fullMsg,JSON.stringify(_chatHistory),JSON.stringify(currentImages));
    var o=JSON.parse(r);setTyping(false);
    if(o.error)addMsg('system','Error: '+o.error);
    else{
      var toolGroupId=null,pendingTools=[];
      for(var i=0;i<o.messages.length;i++){
        var m=o.messages[i];
        if(m.type==='tool'){
          if(!toolGroupId)toolGroupId='tg'+Date.now()+Math.random();
          pendingTools.push(m);
        }else{
          addMsg('assistant',m.content,null,null,toolGroupId);_chatHistory.push({type:'assistant',content:m.content});
          for(var j=0;j<pendingTools.length;j++){
            var pt=pendingTools[j];
            addTool(pt.name,pt.args,pt.result,toolGroupId);
          }
          pendingTools=[];toolGroupId=null;
        }
      }
      // Flush any remaining tools (no trailing assistant message)
      for(var j=0;j<pendingTools.length;j++){
        addTool(pendingTools[j].name,pendingTools[j].args,pendingTools[j].result,null);
      }
    }
  }catch(e){setTyping(false);addMsg('system','Error: '+e.message);}
  _quoted=[];updateInputQuote();
  document.getElementById('send-btn').disabled=false;
  document.getElementById('input').focus();
}

var _quoted=[],_images=[];
function addImage(dataUrl,mediaType){
  _images.push({data:dataUrl.split(',')[1],media_type:mediaType||'image/png'});
  renderImagePreview();
}
function removeImage(idx){
  _images.splice(idx,1);
  renderImagePreview();
}
function renderImagePreview(){
  var c=document.getElementById('img-preview');
  if(_images.length===0){c.classList.remove('has-images');c.innerHTML='';return;}
  c.classList.add('has-images');
  var h='';
  for(var i=0;i<_images.length;i++){
    var mt=_images[i].media_type||'image/png';
    h+='<div class="img-thumb"><img src="data:'+mt+';base64,'+_images[i].data+'" alt="img"><button class="rm" onclick="removeImage('+i+')">x</button></div>';
  }
  c.innerHTML=h;
}
function handleFileSelect(e){
  var files=e.target.files;
  for(var i=0;i<files.length;i++){(function(file){
    var reader=new FileReader();
    reader.onload=function(ev){
      var dataUrl=ev.target.result;
      var mt=file.type||'image/png';
      addImage(dataUrl,mt);
    };
    reader.readAsDataURL(file);
  })(files[i]);}
  e.target.value='';
}
document.addEventListener('paste',function(e){
  var items=e.clipboardData&&e.clipboardData.items;
  if(!items)return;
  for(var i=0;i<items.length;i++){
    if(items[i].type.indexOf('image/')===0){
      e.preventDefault();
      var blob=items[i].getAsFile();
      var reader=new FileReader();
      reader.onload=function(ev){addImage(ev.target.result,blob.type);};
      reader.readAsDataURL(blob);
      break;
    }
  }
});
function addMsg(role,text,images,msgId,toolGroupId){
  var id=msgId||('m'+Date.now()+Math.random());
  var d=document.createElement('div');
  var a={'user':'U','assistant':'AI','system':'!'};
  d.className='msg '+role;
  d.setAttribute('data-msg-id',id);
  d.setAttribute('data-msg-text',text);
  var imgHtml='';
  if(images&&images.length>0){
    for(var i=0;i<images.length;i++){
      var mt=images[i].media_type||'image/png';
      imgHtml+='<img class="msg-img" src="data:'+mt+';base64,'+images[i].data+'" onclick="this.style.maxWidth=this.style.maxWidth==\'600px\'?\'200px\':\'600px\'" title="Click to enlarge">';
    }
  }
  var detailHtml='';
  if(toolGroupId){
    detailHtml='<div class="detail-toggle" data-tool-group="'+toolGroupId+'" onclick="toggleDetail(this)">详细过程 ▼</div>';
  }
  var actions='';
  if(role!=='system'){
    actions='<div class="msg-actions">'+
      (role==='user'?'<button onclick="retryMsg(\''+id+'\')" title="重试">重试</button>':'')+
      '<button onclick="copyMsg(\''+id+'\')" title="复制">复制</button>'+
      '<input type="checkbox" class="quote-cb" onclick="toggleQuote(this,\''+id+'\')" title="Quote">'+
      '</div>';
  }
  d.innerHTML='<div class="avatar '+role+'">'+(a[role]||'?')+'</div><div style="flex:1"><div class="bubble '+role+'">'+esc(text)+imgHtml+'</div>'+actions+detailHtml+'</div>';
  document.getElementById('chat').insertBefore(d,document.getElementById('typing'));
  document.getElementById('chat').scrollTop=document.getElementById('chat').scrollHeight;
  return d;
}

function addTool(name,args,result,groupId){
  var c=document.getElementById('chat'),t=document.getElementById('typing');
  var sa=args.length>120?args.slice(0,120)+'...':args;
  var sr=result.length>400?result.slice(0,400)+'...':result;
  var id1='t'+Date.now()+Math.random(),id2='r'+Date.now()+Math.random();
  var agentBadge='';
  if(name==='run_vision_subagent')agentBadge=' [vision]';
  else if(name==='run_subtask')agentBadge=' [subtask]';
  else if(name==='run_intent_subagent')agentBadge=' [intent]';
  var act='<div class="msg-actions"><button onclick="copyMsg(\''+id1+'\')" title="复制">复制</button><input type="checkbox" class="quote-cb" onclick="toggleQuote(this,\''+id1+'\')" title="Quote"></div>';
  var d1=document.createElement('div');d1.className='msg tool';
  if(groupId){d1.classList.add('tool-collapsed');d1.setAttribute('data-tool-group',groupId);}
  d1.setAttribute('data-msg-id',id1);
  d1.setAttribute('data-msg-text',name+' '+args);
  d1.innerHTML='<div class="avatar tool">T</div><div style="flex:1"><div class="bubble tool">'+esc(name)+agentBadge+'  '+esc(sa)+'</div>'+act+'</div>';
  c.insertBefore(d1,t);
  var act2='<div class="msg-actions"><button onclick="copyMsg(\''+id2+'\')" title="复制">复制</button><input type="checkbox" class="quote-cb" onclick="toggleQuote(this,\''+id2+'\')" title="Quote"></div>';
  var d2=document.createElement('div');d2.className='msg tool-result';
  if(groupId){d2.classList.add('tool-collapsed');d2.setAttribute('data-tool-group',groupId);}
  d2.setAttribute('data-msg-id',id2);
  d2.setAttribute('data-msg-text',result);
  d2.innerHTML='<div class="avatar tool-result">-</div><div style="flex:1"><div class="bubble tool-result">'+esc(sr)+'</div>'+act2+'</div>';
  c.insertBefore(d2,t);
  c.scrollTop=c.scrollHeight;
}

function setTyping(s,status){
  document.getElementById('typing').classList.toggle('active',s);
  document.getElementById('typing-status').textContent=status||'';
}

function copyMsg(id){
  var el=document.querySelector('[data-msg-id="'+id+'"]');
  if(!el)return;
  var text=el.getAttribute('data-msg-text')||el.querySelector('.bubble').textContent;
  // Try JS clipboard first
  var ta=document.createElement('textarea');
  ta.value=text;ta.style.position='fixed';ta.style.left='-9999px';
  document.body.appendChild(ta);ta.select();
  var ok=false;
  try{ok=document.execCommand('copy');}catch(e){}
  document.body.removeChild(ta);
  // Fallback to Python-side clipboard
  if(!ok){try{window.pywebview.api.copy_to_clipboard(text);}catch(e){}}
}

function retryMsg(id){
  var el=document.querySelector('[data-msg-id="'+id+'"]');
  if(!el)return;
  var text=el.getAttribute('data-msg-text');
  if(!text)return;
  document.getElementById('input').value=text;
  // Look for images in this message to restore
  var imgs=el.querySelectorAll('.msg-img');
  _images=[];
  for(var i=0;i<imgs.length;i++){
    var src=imgs[i].src;
    var parts=src.split(',');
    if(parts.length>1)_images.push({data:parts[1],media_type:'image/png'});
  }
  renderImagePreview();
  sendMsg();
}

function toggleQuote(cb,id){
  var el=document.querySelector('[data-msg-id="'+id+'"]');
  if(!el)return;
  var text=el.getAttribute('data-msg-text')||el.querySelector('.bubble').textContent;
  if(cb.checked){
    var imgs=[];
    var imels=el.querySelectorAll('.msg-img');
    for(var i=0;i<imels.length;i++){
      var src=imels[i].src,parts=src.split(',');
      if(parts.length>1)imgs.push({data:parts[1],media_type:'image/png'});
    }
    _quoted.push({id:id,text:text,images:imgs});
  }
  else{_quoted=_quoted.filter(function(q){return q.id!==id;});}
  updateInputQuote();
}

function updateInputQuote(){
  var input=document.getElementById('input');
  var existing=input.value.replace(/^\[已选中\d+条消息\][\s\S]*$/,'').trim();
  if(_quoted.length===0){input.value=existing;return;}
  var prefix='[已选中'+_quoted.length+'条消息] ';
  var previews=[];
  for(var i=0;i<_quoted.length;i++){
    var t=_quoted[i].text.replace(/\n/g,' ').slice(0,40);
    previews.push((i+1)+'. '+t+(t.length>=40?'...':''));
  }
  input.value=prefix+previews.join('  ')+'\n'+existing;
}
async function compress(keep){
  if(!keep){
    try{var c=JSON.parse(await window.pywebview.api.get_config());keep=c.compress_keep||10;}catch(e){keep=10;}
  }
  try{
    var r=await window.pywebview.api.compress_context(keep);
    var o=JSON.parse(r);
    if(o.compressed>0){
      addMsg('system','已压缩 '+o.compressed+' 条旧消息，保留最近 '+o.kept+' 条');
      // Dim old messages visually but keep them in DOM for display/copy/quote
      var ms=document.querySelectorAll('.msg');
      for(var i=0;i<ms.length-o.kept;i++){if(ms[i])ms[i].style.opacity='0.3';}
    }else{
      addMsg('system','仅 '+o.kept+' 条消息，无需压缩');
    }
  }catch(e){addMsg('system','压缩失败: '+e.message);}
}

function clearChat(){
  var ms=document.querySelectorAll('.msg');
  for(var i=0;i<ms.length;i++)ms[i].remove();
  document.getElementById('empty').style.display='flex';
  _chatHistory=[];
  _quoted=[];
  _images=[];renderImagePreview();
  window.pywebview.api.clear_context();
}

async function openSettings(){
  document.getElementById('settings-modal').classList.add('active');
  switchSettingsTab('main');
  try{var r=await window.pywebview.api.get_config();var c=JSON.parse(r);
    var pk='custom';
    if(c.provider==='anthropic')pk='anthropic';
    else if(c.openai_base_url&&c.openai_base_url.indexOf('deepseek')>=0)pk='deepseek';
    else if(c.openai_base_url&&c.openai_base_url.indexOf('dashscope')>=0)pk='qwen';
    else if(c.openai_base_url==='https://api.openai.com/v1')pk='openai';
    document.getElementById('s-prov').value=pk;
    document.getElementById('s-model').value=c.model||'';
    document.getElementById('s-key').value='';
    document.getElementById('s-url').value=c.openai_base_url||'';
    var ip=c.intent_provider||'';
    document.getElementById('s-iprov').value=ip;
    document.getElementById('s-imodel').value=c.intent_model||'';
    document.getElementById('s-ikey').value='';
    document.getElementById('s-iurl').value=c.intent_base_url||'';
    onIProvChange();
    var vp=c.vision_provider||'';
    document.getElementById('s-vprov').value=vp;
    document.getElementById('s-vmodel').value=c.vision_model||'';
    document.getElementById('s-vkey').value='';
    document.getElementById('s-vurl').value=c.vision_base_url||'';
    onVProvChange();
    document.getElementById('s-keep').value=c.compress_keep||10;
    document.getElementById('s-convpath').value=c.conversation_path||'';
    onProvChange();
  }catch(e){}
}
function switchSettingsTab(name){
  var tabs=document.querySelectorAll('.settings-tab');
  for(var i=0;i<tabs.length;i++)tabs[i].classList.toggle('active',tabs[i].id==='tab-'+name);
  var btns=document.querySelectorAll('.settings-sidebar .cat-btn');
  for(var i=0;i<btns.length;i++)btns[i].classList.toggle('active',btns[i].getAttribute('data-tab')===name);
}
function closeSettings(){document.getElementById('settings-modal').classList.remove('active');}

function onProvChange(){
  var k=document.getElementById('s-prov').value,p=PRESETS[k];
  document.getElementById('s-model').value=p.m;
  document.getElementById('s-url').value=p.u;
  document.getElementById('s-url').disabled=p.p==='anthropic';
}

async function saveSettings(){
  closeSettings();
  var k=document.getElementById('s-prov').value,p=PRESETS[k];
  var ik=document.getElementById('s-iprov').value;
  var vk=document.getElementById('s-vprov').value;
  await window.pywebview.api.save_settings(p.p,document.getElementById('s-model').value,document.getElementById('s-key').value,document.getElementById('s-url').value,
    ik,document.getElementById('s-imodel').value,document.getElementById('s-ikey').value,document.getElementById('s-iurl').value,
    vk,document.getElementById('s-vmodel').value,document.getElementById('s-vkey').value,document.getElementById('s-vurl').value,
    document.getElementById('s-keep').value,document.getElementById('s-convpath').value);
  try{var r=await window.pywebview.api.get_config();var c=JSON.parse(r);
    document.getElementById('st-main').textContent='Main: '+(c.model||'?')+' @ '+(c.provider||'?');
    var im=c.intent_model;document.getElementById('st-intent').textContent=im?'| Intent: '+im+' @ '+(c.intent_provider||'?'):'';
    var vm=c.vision_model;document.getElementById('st-vision').textContent=vm?'| Vision: '+vm+' @ '+(c.vision_provider||'?'):'';
  }catch(e){}
}

function onIProvChange(){
  var k=document.getElementById('s-iprov').value;
  if(!k){document.getElementById('s-imodel').value='';document.getElementById('s-iurl').value='';return;}
  var p=IPRESETS[k];
  if(!document.getElementById('s-imodel').value)document.getElementById('s-imodel').value=p.m;
  if(!document.getElementById('s-iurl').value)document.getElementById('s-iurl').value=p.u;
}

function onVProvChange(){
  var k=document.getElementById('s-vprov').value;
  if(!k){document.getElementById('s-vmodel').value='';document.getElementById('s-vurl').value='';return;}
  var p=VPRESETS[k];
  if(!document.getElementById('s-vmodel').value)document.getElementById('s-vmodel').value=p.m;
  if(!document.getElementById('s-vurl').value)document.getElementById('s-vurl').value=p.u;
}

async function openMemories(){
  document.getElementById('memories-modal').classList.add('active');
  try{
    var r=await window.pywebview.api.get_memories();
    var ms=JSON.parse(r),l=document.getElementById('memory-list');
    if(!ms||ms.length===0)l.innerHTML='<li style="color:var(--gray)">No memories stored yet.</li>';
    else{
      var h='';
      for(var i=0;i<ms.length;i++)h+='<li><b style="color:var(--yellow)">['+(ms[i].type||'?')+']</b> '+(ms[i].name||'')+' - '+(ms[i].description||'')+'</li>';
      l.innerHTML=h;
    }
  }catch(e){document.getElementById('memory-list').innerHTML='<li style="color:var(--red)">Failed to load: '+e.message+'</li>';}
}
function closeMemories(){document.getElementById('memories-modal').classList.remove('active');}

async function openSkills(){
  document.getElementById('skills-modal').classList.add('active');
  try{
    var r=await window.pywebview.api.list_skills();
    var sk=JSON.parse(r),l=document.getElementById('skill-list');
    if(!sk||sk.length===0)l.innerHTML='<li style="color:var(--gray)">No skills found.</li>';
    else{
      var h='';
      for(var i=0;i<sk.length;i++)h+='<li><b style="color:var(--green)">'+sk[i].name+'</b> — '+sk[i].description+'</li>';
      l.innerHTML=h;
    }
  }catch(e){document.getElementById('skill-list').innerHTML='<li style="color:var(--red)">Error: '+e.message+'</li>';}
}
function closeSkills(){document.getElementById('skills-modal').classList.remove('active');}
function addNewSkill(){
  var name=prompt('Skill name (e.g. "optimize-sql"):');
  if(!name)return;
  var desc=prompt('Description:');
  if(!desc)return;
  var prompt_text=prompt('Skill prompt (instructions for the AI):');
  if(!prompt_text)return;
  window.pywebview.api.save_skill(name,desc,prompt_text);
  openSkills();
}

async function openMcp(){
  document.getElementById('mcp-modal').classList.add('active');
  await refreshMcp();
}
function closeMcp(){document.getElementById('mcp-modal').classList.remove('active');}
async function refreshMcp(){
  try{
    var r=await window.pywebview.api.list_mcp_servers();
    var sr=JSON.parse(r),l=document.getElementById('mcp-list');
    if(!sr||sr.length===0)l.innerHTML='<li style="color:var(--gray)">No MCP servers configured.</li>';
    else{
      var h='';
      for(var i=0;i<sr.length;i++){
        var s=sr[i],color=s.status==='connected'?'var(--green)':'var(--gray)';
        h+='<li><b style="color:'+color+'">'+s.name+'</b> ['+s.status+'] — '+s.description+'</li>';
      }
      l.innerHTML=h;
    }
  }catch(e){document.getElementById('mcp-list').innerHTML='<li style="color:var(--red)">Error: '+e.message+'</li>';}
}
async function addMcpServer(){
  var name=document.getElementById('mcp-name').value.trim();
  var cmd=document.getElementById('mcp-cmd').value.trim();
  if(!name||!cmd)return;
  var args=document.getElementById('mcp-args').value.trim()||'[]';
  var desc=name;
  await window.pywebview.api.add_mcp_server(name,cmd,args,'{}',desc);
  document.getElementById('mcp-name').value='';
  document.getElementById('mcp-cmd').value='';
  document.getElementById('mcp-args').value='';
  refreshMcp();
}
async function connectMcpAll(){
  var r=await window.pywebview.api.connect_mcp_server();
  refreshMcp();
}

async function openLogs(){
  document.getElementById('logs-modal').classList.add('active');
  await refreshLogs();
}
function closeLogs(){document.getElementById('logs-modal').classList.remove('active');}

function toggleContext(){
  var c=document.getElementById('context-modal');
  if(c.classList.contains('active')){closeContext();return;}
  document.getElementById('ctx-content').innerHTML='<div style="color:var(--gray);padding:12px 0">加载中...</div>';
  c.classList.add('active');
  window.pywebview.api.get_context().then(function(r){
    var msgs=JSON.parse(r),h='';
    if(!msgs||msgs.length===0){h='<div style="color:var(--gray)">(空)</div>';}
    else for(var i=0;i<msgs.length;i++){
      var m=msgs[i],role=m.role||'?';
      var color=role==='system'?'var(--purple)':role==='user'?'var(--blue)':role==='assistant'?'var(--green)':'var(--orange)';
      h+='<div class="ctx-item" style="border-left-color:'+color+'"><b style="color:'+color+'">'+role+'</b>: '+esc(m.content)+'</div>';
    }
    document.getElementById('ctx-content').innerHTML=h||'<div style="color:var(--gray)">(空)</div>';
  }).catch(function(e){
    document.getElementById('ctx-content').innerHTML='<div style="color:var(--red)">加载失败: '+e.message+'</div>';
  });
}
function closeContext(){document.getElementById('context-modal').classList.remove('active');}
function toggleDetail(el){
  var gid=el.getAttribute('data-tool-group');
  var tools=document.querySelectorAll('.msg.tool[data-tool-group="'+gid+'"],.msg.tool-result[data-tool-group="'+gid+'"]');
  if(tools.length===0)return;
  var shown=!tools[0].classList.contains('tool-collapsed');
  for(var i=0;i<tools.length;i++)tools[i].classList.toggle('tool-collapsed',shown);
  el.textContent=shown?'详细过程 ▼':'详细过程 ▲';
}
async function refreshLogs(){
  try{
    var r=await window.pywebview.api.get_logs();
    document.getElementById('log-content').textContent=r||'(empty)';
  }catch(e){document.getElementById('log-content').textContent='Error: '+e.message;}
}

function esc(t){var d=document.createElement('div');d.textContent=t;return d.innerHTML;}

var _permUid=null,_permSelected=0;
function showPermissionDialog(payload){
  _permUid=payload.uid;_permSelected=0;
  var d=document.getElementById('perm-overlay');
  var isDanger=payload.risk_level==='dangerous';
  document.getElementById('perm-icon').textContent=isDanger?'⚠️':'ℹ️';
  var title=(payload.action_name||payload.tool_name)+(isDanger?' — 危险操作':'');
  if(payload.explain)title+='\n'+payload.explain;
  document.getElementById('perm-title').textContent=title;
  document.getElementById('perm-section-label').textContent=payload.tool_name==='run_bash'?'原生命令：':'此操作将：';
  document.getElementById('perm-msg').textContent=payload.message||payload.tool_name;
  var opts,container=document.getElementById('perm-options');
  if(isDanger){
    opts=[
      {label:'允许 (Allow)', result:'allow', hint:''},
      {label:'拒绝 (Deny)', result:'deny', hint:''},
    ];
  }else{
    opts=[
      {label:'允许本次 (Allow)', result:'allow', hint:'仅此次操作'},
      {label:'本轮全部允许 (Allow All)', result:'allow_all', hint:'该轮对话中不再询问此工具'},
      {label:'拒绝 (Deny)', result:'deny', hint:''},
    ];
  }
  var h='';
  for(var i=0;i<opts.length;i++){
    h+='<div class="perm-opt'+(i===0?' selected':'')+'" data-idx="'+i+'" onclick="_permSelect('+i+');_permConfirm()">'+
      '<span class="opt-key">'+(i+1)+'</span>'+
      '<span class="opt-label">'+esc(opts[i].label)+'</span>'+
      (opts[i].hint?'<span class="opt-hint">'+esc(opts[i].hint)+'</span>':'')+
      '</div>';
  }
  container.innerHTML=h;
  _permOpts=opts;
  d.classList.add('active');
}
var _permOpts=[];
function _permSelect(idx){
  _permSelected=idx;
  var els=document.querySelectorAll('.perm-opt');
  for(var i=0;i<els.length;i++)els[i].classList.toggle('selected',i===idx);
}
function _permConfirm(){
  if(_permUid===null)return;
  var result=_permOpts[_permSelected]?_permOpts[_permSelected].result:'deny';
  document.getElementById('perm-overlay').classList.remove('active');
  try{window.pywebview.api.permission_response(_permUid,result);}catch(e){}
  _permUid=null;
}
function _permDeny(){
  if(_permUid===null)return;
  document.getElementById('perm-overlay').classList.remove('active');
  try{window.pywebview.api.permission_response(_permUid,'deny');}catch(e){}
  _permUid=null;
}

document.addEventListener('keydown',function(e){
  // Permission dialog keyboard handling
  if(_permUid!==null){
    if(e.key==='ArrowDown'||e.key==='ArrowUp'){
      e.preventDefault();e.stopPropagation();
      var n=_permOpts.length;
      _permSelect(e.key==='ArrowDown'?(_permSelected+1)%n:(_permSelected-1+n)%n);
      return;
    }
    if(e.key==='Enter'&&(e.ctrlKey||e.shiftKey)){
      e.preventDefault();e.stopPropagation();
      _permConfirm();
      return;
    }
    if(e.key==='Escape'){
      e.preventDefault();e.stopPropagation();
      _permDeny();
      return;
    }
    if(e.key>='1'&&e.key<='9'){
      var idx=parseInt(e.key)-1;
      if(idx<_permOpts.length){e.preventDefault();e.stopPropagation();_permSelect(idx);}
    }
  }
  if(e.key==='Enter'){
    if(e.ctrlKey||e.shiftKey){
      e.preventDefault();
      var ta=document.getElementById('input');
      var s=ta.selectionStart,end=ta.selectionEnd;
      ta.value=ta.value.slice(0,s)+'\n'+ta.value.slice(end);
      ta.selectionStart=ta.selectionEnd=s+1;
      ta.dispatchEvent(new Event('input'));
    }else{
      e.preventDefault();
      sendMsg();
    }
  }
  if(e.key==='Escape'){closeSettings();closeMemories();closeLogs();closeSkills();closeMcp();closeContext();}
});
</script>
</body>
</html>"""


def start():
    api = BackendAPI()
    webview.create_window(
        title="LocalCode",
        html=HTML,
        js_api=api,
        width=1000,
        height=700,
        min_size=(700, 450),
    )
    webview.start(gui=None)


if __name__ == "__main__":
    start()
