"""1 件 1 プロセスの実行器: `python -m src.train <json>`。本体は imported module の src.react（観測で __main__ にならないように）。"""

import json
import sys

from src.react import solve_instance

if __name__ == "__main__":
    solve_instance(json.loads(sys.argv[1]))
