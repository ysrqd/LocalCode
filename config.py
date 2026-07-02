import os
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".localcode"
CONFIG_FILE = CONFIG_DIR / "config.json"

PROVIDER_PRESETS = {
    "anthropic": {
        "name": "Anthropic (Claude)",
        "provider": "anthropic",
        "default_model": "claude-sonnet-4-6",
        "default_url": "",
    },
    "openai": {
        "name": "OpenAI (GPT)",
        "provider": "openai",
        "default_model": "gpt-4o",
        "default_url": "https://api.openai.com/v1",
    },
    "deepseek": {
        "name": "DeepSeek",
        "provider": "openai",
        "default_model": "deepseek-chat",
        "default_url": "https://api.deepseek.com/v1",
    },
    "qwen": {
        "name": "通义千问 (Qwen)",
        "provider": "openai",
        "default_model": "qwen-plus",
        "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

DEFAULT_CONFIG = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "anthropic_api_key": "",
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "max_tokens": 8192,
    "workspace": "",
    # Vision API (falls back to main provider if not set)
    "vision_provider": "",
    "vision_model": "",
    "vision_api_key": "",
    "vision_base_url": "",
    # Intent recognition API (falls back to main provider if not set)
    "intent_provider": "",
    "intent_model": "",
    "intent_api_key": "",
    "intent_base_url": "",
    "compress_keep": 10,
    "conversation_path": "",
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return {**DEFAULT_CONFIG, **data}
    return {**DEFAULT_CONFIG}


def save_config(cfg: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def get_api_key(provider: str) -> str:
    cfg = load_config()
    if provider == "anthropic":
        return cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    else:
        return cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")


def get_vision_config() -> dict | None:
    """Return vision provider config if configured, else None (use main provider)."""
    cfg = load_config()
    vp = cfg.get("vision_provider", "")
    if not vp:
        return None
    return {
        "provider": vp,
        "model": cfg.get("vision_model", ""),
        "api_key": cfg.get("vision_api_key", "") or get_api_key(vp),
        "base_url": cfg.get("vision_base_url", ""),
    }


def get_intent_config() -> dict | None:
    """Return intent recognition provider config if configured, else None."""
    cfg = load_config()
    ip = cfg.get("intent_provider", "")
    if not ip:
        return None
    return {
        "provider": ip,
        "model": cfg.get("intent_model", ""),
        "api_key": cfg.get("intent_api_key", "") or get_api_key(ip),
        "base_url": cfg.get("intent_base_url", ""),
    }
