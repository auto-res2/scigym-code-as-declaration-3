"""1 つの run（= 1 モデル × 1 split）で SciGym の各件を公式 Controller（ReAct）に解かせ、評価層 scigym_<split> の入力ファイルを書く。

件ごとに 1 プロセス（src/runner.py → src.train → src.react）。モデルは Vercel AI Gateway 経由。
run の設定は config/run/<run_id>.yaml にあり、キーは値を消費する関数の引数名（src/keys.py）。
"""

import gzip
import json
import shutil
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from src import keys as K
from src.preprocess import instances
from src.runner import BUDGET_EXCEEDED, run_all


def cli_args():
    """hydra 形式の key=value を読む。hydra / omegaconf は scigym が固定する petab の antlr 版と衝突する"""
    return {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}


def main():
    cli = cli_args()
    run_id = cli["run"]
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg.update(cli)
    cfg["run"] = run = yaml.safe_load(open(f"config/run/{run_id}.yaml"))  # CLI の run=<run_id> を yaml の中身で置き換える
    cfg = SimpleNamespace(**cfg, task=f"scigym_{run[K.SPLIT]}")
    run_dir = Path(cfg.results_dir) / run_id
    todo = instances(cfg.data_root, run[K.SPLIT], cfg.mode, run.get(K.N_INSTANCES))
    stage = cfg.mode.upper()
    # 打ち切った前 run の件ごとの書庫（リポジトリの resume/<run_id>/）から続きを実行する
    if not (run_dir / "instances").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        for archive in sorted(Path("resume", run_id).glob("**/*.tar.gz")):
            with tarfile.open(archive) as tar:
                tar.extractall(run_dir)
    # API エラーで tokens.json が出なかった件は 2 回までやり直す。予算超過（402）は即座に run を止める
    for _ in range(3):
        codes = run_all(todo, run_dir, cfg, run[K.WORKERS], run[K.INSTANCE_TIMEOUT], run[K.STALL_TIMEOUT])
        if BUDGET_EXCEEDED in codes:
            print(f"{stage}_VALIDATION: FAIL reason=budget_exceeded")
            sys.exit(1)
    submitted, totals = {}, {"input_tokens": 0, "output_tokens": 0}
    n_experiments, n_overflow = [], 0
    for instance in todo:
        out = run_dir / "instances" / instance.name
        if (out / "context_overflow").exists():
            n_overflow += 1
        if not (out / "tokens.json").exists():
            continue  # 文脈超過か、3 試行とも終わらなかった件: 提出無しとして評価層に渡す
        info = json.loads((out / "tokens.json").read_text())
        submitted[instance.name] = (out / "final_model.xml").read_text() if (out / "final_model.xml").exists() else None
        n_experiments.append(info["n_experiments"])
        for k in totals:
            totals[k] += info[k]
    totals |= {
        "mean_experiments": sum(n_experiments) / len(n_experiments) if n_experiments else 0.0,
        "n_with_submission": sum(v is not None for v in submitted.values()),
        "n_context_overflow": n_overflow,
        "n_not_finished": len(todo) - len(submitted),
    }
    (run_dir / "eval_inputs").mkdir(parents=True, exist_ok=True)
    # 評価入力は gzip で書く（参照 SBML と提出 SBML を全件持つので大きい）
    with gzip.open(run_dir / "eval_inputs" / f"{cfg.task}.json.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"instances": [{
            "id": p.name,
            "reference_sbml": (p / "truth.xml").read_text(),
            "incomplete_sbml": (p / "partial.xml").read_text(),
            "reference_sedml": (p / "truth.sedml").read_text(),
            "submitted_sbml": submitted.get(p.name),
        } for p in todo]}))
    (run_dir / "tokens.json").write_text(json.dumps(totals))
    # 取り込みは 1 ファイル 1 API 呼び出しなので、件ごとの出力は 1 つの書庫にまとめる（GitHub の secondary rate limit 対策）
    with tarfile.open(run_dir / "instances.tar.gz", "w:gz") as tar:
        tar.add(run_dir / "instances", arcname="instances")
    shutil.rmtree(run_dir / "instances")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)  # 全件そろったので件ごとの書庫は不要
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps({'n_instances': len(todo), **totals})}")
    if totals["n_with_submission"] == 0:
        print(f"{stage}_VALIDATION: FAIL reason=no_submission")
        sys.exit(1)
    print(f"{stage}_VALIDATION: PASS")


if __name__ == "__main__":
    main()
