"""Skill library — markdown files with prompts. Stored in ~/.localcode/skills/"""

import json
from pathlib import Path

SKILLS_DIR = Path.home() / ".localcode" / "skills"

BUILTIN = {
    "code-review": {
        "name": "code-review",
        "description": "Review code for bugs, security issues, and improvements",
        "prompt": "You are a code reviewer. Analyze the following code carefully. Find bugs, security vulnerabilities, performance issues, and suggest improvements. Be specific — mention line numbers and explain why each issue matters.",
    },
    "explain-code": {
        "name": "explain-code",
        "description": "Explain what a piece of code does in plain language",
        "prompt": "You are a code explainer. Explain the following code clearly and concisely. Describe what it does, how it works, and any key patterns or techniques used. Use plain language suitable for a developer learning the codebase.",
    },
    "refactor": {
        "name": "refactor",
        "description": "Suggest refactoring improvements for cleaner, more maintainable code",
        "prompt": "You are a refactoring expert. Examine the following code and suggest refactoring improvements. Focus on readability, maintainability, DRY principle, and modern best practices. Provide concrete before/after examples.",
    },
    "write-tests": {
        "name": "write-tests",
        "description": "Generate unit tests for the given code",
        "prompt": "You are a test engineer. Write comprehensive unit tests for the following code. Cover edge cases, error paths, and happy paths. Use the testing framework appropriate for the language. Include setup/teardown if needed.",
    },
    "debug-error": {
        "name": "debug-error",
        "description": "Analyze an error message or stack trace and suggest fixes",
        "prompt": "You are a debugging expert. Analyze the following error message or stack trace. Identify the root cause, explain what went wrong, and provide a step-by-step fix. Be specific about which files and lines need to change.",
    },
}


def _ensure_dir():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    # Write builtin skills if they don't exist
    for slug, skill in BUILTIN.items():
        path = SKILLS_DIR / f"{slug}.json"
        if not path.exists():
            path.write_text(json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8")


def list_skills() -> list[dict]:
    """Return list of all available skills."""
    _ensure_dir()
    skills = []
    for f in sorted(SKILLS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            skills.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "file": str(f),
            })
        except Exception:
            pass
    return skills


def get_skill(name: str) -> dict | None:
    """Get a skill by name. Returns {name, description, prompt} or None."""
    _ensure_dir()
    path = SKILLS_DIR / f"{name}.json"
    if not path.exists():
        # Try fuzzy match
        for f in SKILLS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("name") == name:
                    return data
            except Exception:
                pass
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_skill(name: str, description: str, prompt: str) -> str:
    """Create or update a skill. Returns status message."""
    _ensure_dir()
    slug = name.lower().replace(" ", "-")
    skill = {"name": name, "description": description, "prompt": prompt}
    path = SKILLS_DIR / f"{slug}.json"
    path.write_text(json.dumps(skill, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"Skill '{name}' saved (slug: {slug})"


def delete_skill(name: str) -> str:
    """Delete a skill by name or slug."""
    _ensure_dir()
    # Try exact slug match
    path = SKILLS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
        return f"Skill '{name}' deleted."

    # Try fuzzy match
    for f in SKILLS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("name") == name:
                f.unlink()
                return f"Skill '{name}' deleted."
        except Exception:
            pass
    return f"Skill '{name}' not found."
