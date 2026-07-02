"""LocalCode — simplified Claude Code CLI.

Usage:
    localcode                     CLI interactive mode
    localcode --web               Web GUI mode (pywebview)
    localcode --gui               Tkinter GUI mode
    localcode -c "prompt"        One-shot mode
    localcode --config           Configure API keys
"""

import sys
import os


def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    if "--web" in sys.argv or "-w" in sys.argv:
        from gui_web import start
        start()
    elif "--gui" in sys.argv or "-g" in sys.argv:
        from gui import run_gui
        run_gui()
    elif len(sys.argv) > 1:
        if sys.argv[1] == "--config":
            from cli import setup_config
            setup_config()
        elif sys.argv[1] in ("-c", "--command"):
            if len(sys.argv) < 3:
                print("Usage: localcode -c \"your prompt\"")
                sys.exit(1)
            from cli import run_one_shot
            run_one_shot(sys.argv[2])
        elif sys.argv[1] in ("-h", "--help"):
            print(__doc__)
        else:
            from cli import run_one_shot
            run_one_shot(" ".join(sys.argv[1:]))
    else:
        from cli import run_interactive
        run_interactive()


if __name__ == "__main__":
    main()
