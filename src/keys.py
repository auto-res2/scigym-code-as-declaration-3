"""config/run/<run_id>.yaml のキー。

キーは「その値を消費する関数の引数名」（module.Class.method.arg）で、record の params にも同じキーで載る。
フックはその関数の引数を観測し、gate が yaml の値と比べるので、値は変換せずにその引数へ渡す。
"""

# ---------------------------------------------------------------- run 全体（src/runner.py、src/preprocess.py）
WORKERS = "src.runner.run_all.workers"
INSTANCE_TIMEOUT = "src.runner.run_all.instance_timeout"
STALL_TIMEOUT = "src.runner.run_all.stall_timeout"
SPLIT = "src.preprocess.instances.split"
N_INSTANCES = "src.preprocess.instances.n_instances"

# ---------------------------------------------------------------- 上流（SciGym、openai）の引数
MAX_ITERATIONS = "scigym.controller.Controller.__init__.max_iterations"
EVAL_DEBUG_ROUNDS = "scigym.controller.Controller.__init__.eval_debug_rounds"
TEMPERATURE = "scigym.controller.Controller.__init__.temperature"
MODEL = "scigym.api.api.LLM.__init__.model_name"
LLM_TEMPERATURE = "scigym.api.api.LLM.__init__.temperature"
MAX_LENGTH = "scigym.api.api.LLM.__init__.max_length"
BASE_URL = "openai._client.OpenAI.__init__.base_url"
