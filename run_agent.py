"""便捷入口：``python run_agent.py ...`` 等价于 ``python -m agent ...``。"""

from __future__ import annotations

import sys

from agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
