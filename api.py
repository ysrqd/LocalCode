"""LLM client — supports Anthropic and OpenAI-compatible APIs."""

import json

import config
import tools


class LLMClient:
    """Unified client. Uses tools.registry to fetch schemas in the right format."""

    def __init__(self):
        self.provider = config.load_config().get("provider", "anthropic")
        self._setup_client()

    def _setup_client(self):
        cfg = config.load_config()
        self.provider = cfg.get("provider", "anthropic")
        if self.provider == "anthropic":
            api_key = config.get_api_key("anthropic")
            if not api_key:
                self._client = None
                self._error = "ANTHROPIC_API_KEY not set. Run: localcode --config"
            else:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                self._model = cfg.get("model", "claude-sonnet-4-6")
                self._error = None
                self._is_anthropic = True
        else:
            api_key = config.get_api_key("openai")
            base_url = cfg.get("openai_base_url", "https://api.openai.com/v1")
            if not api_key:
                self._client = None
                self._error = "OPENAI_API_KEY not set. Run: localcode --config"
            else:
                import openai
                self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
                self._model = cfg.get("model", "gpt-4o")
                self._error = None
                self._is_anthropic = False

    def chat(self, messages: list[dict], _tool_defs=None) -> tuple[str, list[dict]]:
        """Send a chat request. Returns (text_response, tool_calls)."""
        import logger
        if self._client is None:
            logger.warn("chat called but no client configured")
            return self._error or "Client not configured", []

        last_msg = messages[-1].get("content", "") if messages else ""
        preview = (last_msg if isinstance(last_msg, str) else str(last_msg)[:80]) if messages else ""
        logger.info(f"chat -> {self._model} ({len(messages)} msgs, last: {preview})")

        try:
            if self._is_anthropic:
                return self._chat_anthropic(messages)
            else:
                return self._chat_openai(messages)
        except Exception as e:
            logger.error(f"chat failed: {e}")
            raise

    def describe_images(self, images: list[dict], user_text: str = "") -> str:
        """Single-turn vision call. Returns a text description of the images.
        images is [{"media_type": "image/png", "data": "<base64>"}, ...].
        Falls back to main provider if no vision provider configured."""
        import logger
        vcfg = config.get_vision_config()
        if vcfg is None:
            logger.info("no vision provider configured, using main provider")
            return self._describe_with_main(images, user_text)

        vprovider = vcfg["provider"]
        vmodel = vcfg["model"]
        vkey = vcfg["api_key"]
        vurl = vcfg["base_url"]

        if not vkey:
            logger.warn("vision API key not set, using main provider")
            return self._describe_with_main(images, user_text)

        if user_text:
            prompt = f"Analyze this image and extract only what's relevant to: \"{user_text}\". Describe what you see literally. Do NOT answer the question or give advice — only describe the relevant visual content."
        else:
            prompt = "Extract and describe everything visible in this image — text, code, UI elements, errors, diagrams, etc. Be thorough and literal. Do NOT answer questions or give advice, just describe what you see."

        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.get("media_type", "image/png"), "data": img["data"]},
            })
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        logger.info(f"describe_images -> {vmodel} ({len(images)} images)")

        try:
            if vprovider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=vkey)
                response = client.messages.create(
                    model=vmodel, messages=messages,
                    max_tokens=1024,
                )
                return response.content[0].text
            else:
                import openai
                client = openai.OpenAI(api_key=vkey, base_url=vurl)
                openai_msgs = self._convert_to_openai_messages(messages)
                response = client.chat.completions.create(
                    model=vmodel, messages=openai_msgs,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"vision describe failed: {e}, falling back to main provider")
            return self._describe_with_main(images, user_text)

    def _describe_with_main(self, images: list[dict], user_text: str = "") -> str:
        """Fallback: use main provider to describe images."""
        import logger
        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img.get("media_type", "image/png"), "data": img["data"]},
            })
        if user_text:
            prompt = f"Analyze this image and extract only what's relevant to: \"{user_text}\". Describe what you see literally. Do NOT answer the question or give advice — only describe the relevant visual content."
        else:
            prompt = "Extract and describe everything visible in this image — text, code, UI elements, errors, diagrams, etc. Be thorough and literal. Do NOT answer questions or give advice, just describe what you see."
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        logger.info(f"describe_images (main fallback) -> {self._model}")
        try:
            if self._is_anthropic:
                response = self._client.messages.create(
                    model=self._model, messages=messages,
                    max_tokens=1024,
                )
                return response.content[0].text
            else:
                openai_msgs = self._convert_to_openai_messages(messages)
                response = self._client.chat.completions.create(
                    model=self._model, messages=openai_msgs,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"describe_images fallback failed: {e}")
            return f"(image description failed: {e})"

    def _chat_anthropic(self, messages: list[dict]) -> tuple[str, list[dict]]:
        cfg = config.load_config()
        system = ""
        anthropic_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                anthropic_msgs.append(m)

        # Wrap universal schemas as Anthropic expects: {name, description, input_schema}
        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools.registry.to_schemas()
        ]

        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=anthropic_msgs,
            max_tokens=cfg.get("max_tokens", 8192),
            tools=anthropic_tools,
        )

        text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return text, tool_calls

    def _chat_openai(self, messages: list[dict]) -> tuple[str, list[dict]]:
        cfg = config.load_config()
        openai_msgs = self._convert_to_openai_messages(messages)

        # Wrap universal schemas as OpenAI expects: {type: "function", function: {name, description, parameters}}
        openai_tools = [
            {"type": "function", "function": t}
            for t in tools.registry.to_schemas()
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=openai_msgs,
            max_tokens=cfg.get("max_tokens", 8192),
            tools=openai_tools if openai_tools else None,
        )

        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return text, tool_calls

    def _convert_to_openai_messages(self, messages: list[dict]) -> list[dict]:
        """Convert Anthropic-format messages to OpenAI format."""
        openai_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]

            if role == "system":
                openai_msgs.append({"role": "system", "content": content})
            elif role == "user":
                if isinstance(content, list):
                    openai_content = []
                    has_tool_results = False
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            openai_msgs.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                            has_tool_results = True
                        elif isinstance(block, dict) and block.get("type") == "text":
                            openai_content.append({"type": "text", "text": block["text"]})
                        elif isinstance(block, dict) and block.get("type") == "image":
                            source = block.get("source", {})
                            data = source.get("data", "")
                            media_type = source.get("media_type", "image/png")
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{data}"},
                            })
                    if openai_content and not has_tool_results:
                        openai_msgs.append({"role": "user", "content": openai_content})
                    elif openai_content:
                        openai_msgs.append({"role": "user", "content": openai_content})
                else:
                    openai_msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                if isinstance(content, list):
                    text = ""
                    tool_calls = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block["text"]
                        elif isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                                },
                            })
                    msg = {"role": "assistant", "content": text or None}
                    if tool_calls:
                        msg["tool_calls"] = tool_calls
                    openai_msgs.append(msg)
                else:
                    openai_msgs.append({"role": "assistant", "content": content})
        return openai_msgs


def get_client() -> LLMClient:
    return LLMClient()
