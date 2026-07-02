import json
import tools as tool_mod
from api import get_client, LLMClient
import memory
import config

MAX_TOOL_ROUNDS = 15
SUBAGENT_ROUNDS = 5

_BASE_SYSTEM = """You are localcode, a CLI coding assistant. You help users with software engineering tasks.

You have access to tools: {tool_list}.
Use them to read, write, and edit code, search the codebase, run shell commands, and manage persistent memory.

Guidelines:
- Be concise and direct. No fluff.
- When editing files, use edit_file with the exact string to replace.
- Prefer reading files before editing them.
- Only use run_bash for shell commands.
- Write idiomatic, clean code.
- Don't add unnecessary comments.
- Use save_memory when the user shares preferences, project context, or anything worth remembering across sessions.
- Use list_memories to check what's already saved before writing new memories.
"""

_SUBAGENT_NOTE = """
You have access to subagents for complex tasks:
- If the intent analysis suggests sub-questions, use run_subtask for each one. The subagent has the same tools as you. Collect results and synthesize a final answer.
- If the user message mentions images, screenshots, pictures, or attachments, you MUST call run_vision_subagent FIRST before responding. The images are stored on the server and you cannot see them directly. Even if the intent analysis says "no image provided", trust the user message over the intent analysis.
Always decide yourself whether to delegate — don't over-split simple tasks.
"""

# Module-level ref for tools to access the current agent
_current_agent: "Agent | None" = None


def _build_system_prompt() -> str:
    tool_names = ", ".join(tool_mod.registry.list_names())
    prompt = _BASE_SYSTEM.format(tool_list=tool_names)
    prompt += _SUBAGENT_NOTE
    # Memory index always visible (cheap, model knows what's available)
    idx = memory.get_index_prompt()
    if idx:
        prompt += "\n" + idx
    return prompt


class Agent:
    def __init__(self, workspace: str = ""):
        self.workspace = workspace
        self.client = get_client()
        self.pending_images: list[dict] = []
        self.messages: list[dict] = [
            {"role": "system", "content": _build_system_prompt()}
        ]

    def run_with_images(self, text: str, images: list[dict] | None = None) -> str:
        """Process a message with optional images (base64). images=[{media_type, data}]"""
        content = []
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.get("media_type", "image/png"), "data": img["data"]},
                })
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})
        return self._run_loop()

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        return self._run_loop()

    def _run_loop(self) -> str:
        for _ in range(MAX_TOOL_ROUNDS):
            text, tool_calls = self.client.chat(self.messages)

            if tool_calls:
                anthropic_content = []
                if text:
                    anthropic_content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                self.messages.append({"role": "assistant", "content": anthropic_content})

                tool_results = []
                for tc in tool_calls:
                    result = tool_mod.registry.execute(tc["name"], tc["arguments"])
                    tool_results.append({
                        "name": tc["name"],
                        "args": tc["arguments"],
                        "result": result,
                    })
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": result,
                            }
                        ],
                    })

                # Show tool usage to user
                for tr in tool_results:
                    print(f"\n  [tool] {tr['name']}({json.dumps(tr['args'], ensure_ascii=False)})")
                    short = tr["result"][:300]
                    if len(tr["result"]) > 300:
                        short += "..."
                    print(f"  [result] {short}")

                if text:
                    return text
            else:
                self.messages.append({"role": "assistant", "content": text})
                return text

        return "(max tool rounds reached)"

    def run_intent_subagent(self, user_text: str) -> str:
        """Analyze user intent before main agent processes. Single-turn, no tools.
        Returns a structured intent analysis."""
        import logger
        if not user_text.strip():
            return ""

        icfg = config.get_intent_config()
        if icfg:
            client = self._make_client(icfg["provider"], icfg["model"], icfg["api_key"], icfg["base_url"])
            logger.info(f"intent subagent -> {icfg['model']}")
        else:
            client = self.client
            logger.info(f"intent subagent (main) -> {client._model}")

        prompt = f"""Analyze the following user query. Return a concise analysis covering:

1. **Clarity**: Is the question clear? What's ambiguous?
2. **Intent**: What is the user really trying to accomplish?
3. **Sub-questions**: If this is complex, list self-contained sub-questions that can be delegated via run_subtask. Mark as "Sub-questions:" followed by numbered items. If simple, write "None".
4. **Rewrite**: If the query is unclear, suggest a clearer version.
5. **Complexity**: simple / medium / complex.

User query: "{user_text}"

Be brief — 5 lines max."""

        msgs = [{"role": "user", "content": prompt}]
        try:
            text, _ = client.chat(msgs)
            return text or ""
        except Exception as e:
            logger.error(f"intent subagent failed: {e}")
            return ""

    def select_memories(self, user_text: str) -> str:
        """Use intent model to select relevant memories. Returns prompt fragment or empty string."""
        import logger
        icfg = config.get_intent_config()
        if icfg:
            client = self._make_client(icfg["provider"], icfg["model"], icfg["api_key"], icfg["base_url"])
        else:
            client = self.client
        try:
            result = memory.select_relevant(user_text, client)
            if result:
                logger.info(f"memory selected: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"memory selection failed: {e}")
            return ""

    def extract_memories_after_turn(self):
        """Auto-extract memories from recent dialogue + consolidate if needed."""
        import logger
        icfg = config.get_intent_config()
        if icfg:
            client = self._make_client(icfg["provider"], icfg["model"], icfg["api_key"], icfg["base_url"])
        else:
            client = self.client
        try:
            n = memory.extract_memories(self.messages, client)
            if n > 0:
                # Refresh system prompt with updated index
                self.messages[0] = {"role": "system", "content": _build_system_prompt()}
            memory.consolidate_memories(client)
        except Exception:
            pass

    def _make_client(self, provider: str, model: str, api_key: str, base_url: str):
        """Create a lightweight LLMClient for subagents."""
        client = LLMClient.__new__(LLMClient)
        client.provider = provider
        client._is_anthropic = provider == "anthropic"
        client._model = model
        client._error = None
        if provider == "anthropic":
            import anthropic
            client._client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai
            client._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client

    def run_vision_subagent(self, query: str) -> str:
        """Spawn a vision-capable subagent. Returns its final response."""
        import logger
        vcfg = config.get_vision_config()
        if not self.pending_images:
            return "(no images to analyze)"

        if vcfg:
            vprovider = vcfg["provider"]
            vmodel = vcfg["model"]
            vkey = vcfg["api_key"]
            vurl = vcfg["base_url"]
            logger.info(f"vision subagent -> {vmodel} ({len(self.pending_images)} images)")
        else:
            # Use main provider — hope it can handle images
            cfg = config.load_config()
            vprovider = cfg.get("provider", "anthropic")
            vmodel = cfg.get("model", "claude-sonnet-4-6")
            vkey = config.get_api_key(vprovider)
            vurl = cfg.get("openai_base_url", "")
            logger.info(f"vision subagent (main fallback) -> {vmodel}")

        if not vkey:
            return "(vision API key not configured)"

        client = self._make_client(vprovider, vmodel, vkey, vurl)

        # Build subagent messages
        sub_msgs = [
            {"role": "system", "content": f"You are a vision analysis subagent. Analyze the attached image(s) to answer: {query}. You have access to tools — use them to read files, search code, etc. if needed. Return a concise, thorough answer to the query."}
        ]

        content = []
        for img in self.pending_images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.get("media_type", "image/png"), "data": img["data"]},
            })
        content.append({"type": "text", "text": query})
        sub_msgs.append({"role": "user", "content": content})

        # Run subagent tool loop
        final_text = ""
        for _ in range(SUBAGENT_ROUNDS):
            text, tool_calls = client.chat(sub_msgs)

            if tool_calls:
                anthropic_content = []
                if text:
                    anthropic_content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    anthropic_content.append({
                        "type": "tool_use", "id": tc["id"],
                        "name": tc["name"], "input": tc["arguments"],
                    })
                sub_msgs.append({"role": "assistant", "content": anthropic_content})

                for tc in tool_calls:
                    result = tool_mod.registry.execute(tc["name"], tc["arguments"])
                    sub_msgs.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": result}],
                    })
                    logger.info(f"[vision sub] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)[:100]})")

                if text:
                    final_text = text
            else:
                sub_msgs.append({"role": "assistant", "content": text})
                final_text = text
                break

        return final_text or "(vision subagent returned no result)"

    def run_subtask(self, query: str) -> str:
        """Spawn a subtask subagent with the same capabilities as the main agent.
        Use for parallel processing of sub-questions. Returns final answer."""
        import logger
        logger.info(f"subtask subagent -> {self.client._model} query={query[:80]}")

        sub_msgs = [
            {"role": "system", "content": f"You are a coding subagent. Answer this subtask concisely and thoroughly. You have full tool access.\n\nSubtask: {query}"}
        ]
        sub_msgs.append({"role": "user", "content": query})

        final_text = ""
        for _ in range(SUBAGENT_ROUNDS):
            text, tool_calls = self.client.chat(sub_msgs)

            if tool_calls:
                anthropic_content = []
                if text:
                    anthropic_content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    anthropic_content.append({
                        "type": "tool_use", "id": tc["id"],
                        "name": tc["name"], "input": tc["arguments"],
                    })
                sub_msgs.append({"role": "assistant", "content": anthropic_content})

                for tc in tool_calls:
                    result = tool_mod.registry.execute(tc["name"], tc["arguments"])
                    sub_msgs.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": result}],
                    })
                    logger.info(f"[subtask] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)[:100]})")

                if text:
                    final_text = text
            else:
                sub_msgs.append({"role": "assistant", "content": text})
                final_text = text
                break

        return final_text or "(subtask returned no result)"

    def clear(self):
        self.pending_images = []
        self.messages = [{"role": "system", "content": _build_system_prompt()}]

    def reload_memory(self):
        """Refresh the system prompt with latest memory."""
        self.messages[0] = {"role": "system", "content": _build_system_prompt()}
