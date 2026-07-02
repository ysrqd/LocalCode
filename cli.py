"""LocalCode CLI — a simplified Claude Code."""

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

import config
from agent import Agent

console = Console()

STYLE = Style.from_dict({
    "prompt": "#00ff88 bold",
    "toolbar": "#888888",
})

HISTORY_FILE = Path.home() / ".localcode" / "history.txt"


def setup_config():
    """Interactive configuration wizard."""
    console.print("[bold cyan]LocalCode Configuration[/bold cyan]\n")

    cfg = config.load_config()
    presets = config.PROVIDER_PRESETS

    console.print("Select AI provider:")
    for i, (key, p) in enumerate(presets.items(), 1):
        console.print(f"  [{i}] {p['name']}")
    console.print("  [5] Other (OpenAI-compatible)")

    choice = input("Choice [1-5]: ").strip()
    keys = list(presets.keys())
    if choice == "5":
        cfg["provider"] = "openai"
        cfg["model"] = input("Model: ").strip() or "gpt-4o"
        cfg["openai_api_key"] = input("API Key: ").strip()
        cfg["openai_base_url"] = input("Base URL: ").strip() or "https://api.openai.com/v1"
    elif choice in ("1", "2", "3", "4"):
        key = keys[int(choice) - 1]
        preset = presets[key]
        cfg["provider"] = preset["provider"]
        cfg["model"] = input(f"Model [{preset['default_model']}]: ").strip() or preset["default_model"]
        if preset["provider"] == "anthropic":
            cfg["anthropic_api_key"] = input("API Key: ").strip()
        else:
            cfg["openai_api_key"] = input("API Key: ").strip()
            cfg["openai_base_url"] = input(f"Base URL [{preset['default_url']}]: ").strip() or preset["default_url"]
    else:
        cfg["provider"] = "anthropic"
        cfg["model"] = input("Model [claude-sonnet-4-6]: ").strip() or "claude-sonnet-4-6"
        cfg["anthropic_api_key"] = input("API Key: ").strip()

    config.save_config(cfg)
    console.print(f"\n[green]Config saved to {config.CONFIG_FILE}[/green]")


def run_one_shot(prompt: str):
    """Run a single prompt and exit."""
    agent = Agent()
    try:
        with console.status("[cyan]Thinking...[/cyan]"):
            response = agent.run(prompt)
        console.print(Markdown(response))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def run_interactive():
    """Run the interactive chat loop."""
    console.print(Panel(
        "[bold cyan]LocalCode[/bold cyan] — simplified Claude Code\n"
        "Type [yellow]/help[/yellow] for commands, [yellow]/quit[/yellow] to exit",
        title="Welcome"
    ))

    cfg = config.load_config()
    provider = cfg.get("provider", "anthropic")
    key = config.get_api_key(provider)
    if not key:
        console.print(f"[yellow]⚠ {provider.upper()}_API_KEY not set. Run [bold]/config[/bold] to configure.[/yellow]\n")

    config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=STYLE,
    )

    agent = Agent()

    while True:
        try:
            user_input = session.prompt([
                ("class:prompt", "\n▸ "),
            ]).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            cmd = user_input[1:].strip().lower()
            if cmd == "quit" or cmd == "exit" or cmd == "q":
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "help":
                show_help()
                continue
            elif cmd == "config":
                setup_config()
                agent.client._setup_client()  # reload
                continue
            elif cmd == "clear":
                agent.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif cmd == "status":
                cfg = config.load_config()
                key = config.get_api_key(cfg["provider"])
                console.print(f"  Provider: {cfg['provider']}")
                console.print(f"  Model: {cfg['model']}")
                console.print(f"  API Key: {'✓ set' if key else '✗ not set'}")
                console.print(f"  Messages: {len(agent.messages) - 1}")
                continue
            elif cmd == "logs":
                import logger
                console.print(logger.tail(30))
                continue
            elif cmd == "memory" or cmd == "memories":
                import memory
                entries = memory.list_memories()
                if not entries:
                    console.print("[dim]No memories stored.[/dim]")
                else:
                    console.print("[bold cyan]Persistent Memories:[/bold cyan]")
                    for m in entries:
                        type_colors = {"user": "blue", "project": "yellow", "reference": "green", "feedback": "magenta"}
                        c = type_colors.get(m.get("type", ""), "white")
                        console.print(f"  [[{c}]{m.get('type','?')}[/{c}]] {m.get('name','?')}: {m.get('description','?')}")
                continue
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                continue

        # Normal message — run agent
        try:
            console.print("")  # spacing
            with console.status("[cyan]Thinking...[/cyan]"):
                response = agent.run(user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def show_help():
    console.print("""
[bold cyan]Commands:[/bold cyan]
  [yellow]/help[/yellow]     — Show this help
  [yellow]/config[/yellow]   — Configure API keys and model
  [yellow]/clear[/yellow]    — Clear conversation history
  [yellow]/status[/yellow]   — Show current settings
  [yellow]/memory[/yellow]   — List stored memories
  [yellow]/quit[/yellow]     — Exit

[bold cyan]Tools available:[/bold cyan]
  read_file, write_file, edit_file, glob_files, grep_search, run_bash
  save_memory, list_memories, delete_memory
""")
