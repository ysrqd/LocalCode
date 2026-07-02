"""Persistent memory system — stores user/project context across sessions."""

import json
import time
from pathlib import Path

MEMORY_DIR = Path.home() / ".localcode" / "memory"
INDEX_FILE = MEMORY_DIR / "MEMORY.md"


def ensure_dir():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def list_memories() -> list[dict]:
    """Return all memory entries with their metadata."""
    ensure_dir()
    entries = []
    for f in MEMORY_DIR.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        content = f.read_text(encoding="utf-8", errors="replace")
        meta = _parse_frontmatter(content)
        meta["file"] = str(f)
        meta["stub"] = _extract_stub(content)
        entries.append(meta)
    return entries


def load_index() -> str:
    """Return the MEMORY.md index content."""
    ensure_dir()
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8", errors="replace")
    return ""


def save_memory(name: str, type_: str, description: str, content: str):
    """Create or update a memory file."""
    ensure_dir()
    # Sanitize name
    safe_name = name.lower().replace(" ", "-").replace("/", "-").replace("\\", "-")
    fpath = MEMORY_DIR / f"{safe_name}.md"

    frontmatter = f"""---
name: {safe_name}
description: {description}
type: {type_}
timestamp: {time.strftime("%Y-%m-%d %H:%M")}
---

{content}
"""
    fpath.write_text(frontmatter, encoding="utf-8")
    _update_index(safe_name, description, type_)


def delete_memory(name: str):
    """Delete a memory by name slug."""
    fpath = MEMORY_DIR / f"{name}.md"
    if fpath.exists():
        fpath.unlink()
    _rebuild_index()


def get_context_prompt() -> str:
    """Build a context string with ALL memories (legacy — use select_relevant instead)."""
    ensure_dir()
    memories = list_memories()
    if not memories:
        return ""

    lines = ["\n\n## Persistent Memory"]
    lines.append("The following information persists across sessions:\n")
    for m in memories:
        lines.append(f"- [{m.get('name','?')}] ({m.get('type','?')}): {m.get('description','')}")
        stub = m.get("stub", "")
        if stub:
            lines.append(f"  {stub[:200]}")
    return "\n".join(lines)


def select_relevant(user_text: str, client, max_items: int = 5) -> str:
    """Use LLM to select memories relevant to the current query. Returns prompt fragment."""
    import json as _json
    import re as _re

    ensure_dir()
    memories = list_memories()
    if not memories:
        return ""

    if len(memories) <= max_items:
        return _format_memories(memories)

    # Build catalog
    catalog_parts = []
    for i, m in enumerate(memories):
        stub = (m.get("stub", "") or "")[:100]
        catalog_parts.append(f"{i}: {m['name']} ({m.get('type','?')}) — {m.get('description','?')}")
        if stub:
            catalog_parts.append(f"   {stub}")

    catalog = "\n".join(catalog_parts)

    prompt = f"""Select up to {max_items} relevant memory indices. Return JSON array only.

User query: "{user_text[:300]}"

Memory catalog:
{catalog}

Return: [0, 3, ...]"""

    try:
        msgs = [{"role": "user", "content": prompt}]
        text, _ = client.chat(msgs)
        text = (text or "").strip()
        match = _re.search(r'\[.*?\]', text)
        if match:
            indices = _json.loads(match.group())
            selected = [memories[i] for i in indices if 0 <= i < len(memories)]
            return _format_memories(selected)
    except Exception:
        pass

    # Fallback: return first few
    return _format_memories(memories[:max_items])


def _format_memories(memories: list[dict]) -> str:
    """Format selected memories for system prompt injection."""
    if not memories:
        return ""
    lines = ["\n\n## Relevant Memory"]
    for m in memories:
        lines.append(f"- [{m.get('name','?')}] ({m.get('type','?')}): {m.get('description','?')}")
        stub = m.get("stub", "")
        if stub:
            lines.append(f"  {stub[:300]}")
    return "\n".join(lines)


# ── Internal helpers ──

def _parse_frontmatter(content: str) -> dict:
    meta = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            for line in content[3:end].strip().split("\n"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    meta[key] = val
    return meta


def _extract_stub(content: str) -> str:
    """Extract body text after frontmatter."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            return content[end + 3:].strip()
    return content.strip()


def _update_index(name: str, description: str, type_: str):
    """Add or update an entry in MEMORY.md."""
    ensure_dir()
    lines = []
    if INDEX_FILE.exists():
        lines = INDEX_FILE.read_text(encoding="utf-8").strip().split("\n")

    new_line = f"- [{name}]({name}.md) — {description} ({type_})"

    # Replace existing entry or append
    found = False
    for i, line in enumerate(lines):
        if f"[{name}]" in line:
            lines[i] = new_line
            found = True
            break

    if not found:
        lines.append(new_line)

    INDEX_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _rebuild_index():
    """Rebuild index from existing memory files."""
    ensure_dir()
    memories = list_memories()
    lines = ["# Memory Index\n"]
    for m in memories:
        lines.append(f"- [{m.get('name','?')}]({m.get('name','?')}.md) — {m.get('description','')} ({m.get('type','?')})")
    INDEX_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def get_index_prompt() -> str:
    """MEMORY.md index line — cheap, always in system prompt so model knows what's available."""
    index = load_index()
    if not index:
        return ""
    lines = index.strip().split("\n")
    # Keep under 200 lines as per Claude Code design
    if len(lines) > 200:
        lines = lines[:200]
    return "\n## Memory Index (available memories)\n" + "\n".join(lines)


def extract_memories(messages: list, client) -> int:
    """Extract new memories from recent dialogue. Runs after each AI response.
    Returns number of new memories extracted."""
    import json as _json
    import re as _re
    import logger

    # Collect recent conversation text (last 10 messages)
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict):
                    t = b.get("type", "?")
                    if t == "text":
                        parts.append(b.get("text", "")[:500])
                    elif t == "tool_use":
                        parts.append(f"[tool: {b.get('name','?')}]")
                else:
                    parts.append(str(b)[:200])
            content = " ".join(parts)
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content[:500]}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return 0

    existing = list_memories()
    existing_desc = "\n".join(
        f"- {m.get('name','?')}: {m.get('description','?')}" for m in existing
    ) if existing else "(none)"

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown, include WHY for feedback/project types\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        msgs = [{"role": "user", "content": prompt}]
        text, _ = client.chat(msgs)
        text = (text or "").strip()
        match = _re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return 0
        items = _json.loads(match.group())
        if not items:
            return 0
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                save_memory(name, mem_type, desc, body)
                count += 1
        if count:
            logger.info(f"extracted {count} new memories")
        return count
    except Exception as e:
        logger.error(f"memory extraction failed: {e}")
        return 0


CONSOLIDATE_THRESHOLD = 10


def consolidate_memories(client) -> int:
    """Merge duplicate/stale memories. Returns new count."""
    import json as _json
    import re as _re
    import logger

    files = list_memories()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return len(files)

    catalog_parts = []
    for f in files:
        catalog_parts.append(
            f"## {f.get('name','?')}\n"
            f"type: {f.get('type','?')}\n"
            f"description: {f.get('description','?')}\n"
            f"{f.get('stub','')[:500]}"
        )
    catalog = "\n\n".join(catalog_parts)

    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )

    try:
        msgs = [{"role": "user", "content": prompt}]
        text, _ = client.chat(msgs)
        text = (text or "").strip()
        match = _re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return len(files)
        items = _json.loads(match.group())

        # Remove old memory files
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                save_memory(name, mem_type, desc, body)

        logger.info(f"consolidated {len(files)} → {len(items)} memories")
        return len(items)
    except Exception as e:
        logger.error(f"memory consolidation failed: {e}")
        return len(files)
