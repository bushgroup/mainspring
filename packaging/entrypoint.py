"""PyInstaller's entry script.

PyInstaller freezes a script, not a console-script name from `pyproject.toml`, so the
`.exe` needs a real file that calls `mainspring.viewer.app.main` -- this one, named
directly by `mainspring.spec`.
"""

import sys

from mainspring.viewer.app import main

if __name__ == "__main__":
    sys.exit(main())
