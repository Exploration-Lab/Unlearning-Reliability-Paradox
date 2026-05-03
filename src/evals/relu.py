from evals.base import Evaluator


class RELUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RELU", eval_cfg, **kwargs)
