from typing import Optional, Dict, Any, List

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .tasks import sample_regr_task
from .prompting import build_prompt
from .llm import llm_predict
from .baselines import gd1_regr_predict


def eval_regr_suite(
    n_tasks: int = 10,
    d: int = 10,
    N: Optional[int] = None,
    alpha: float = 1.0,
    eta: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluate the LLM and the GD-1 baseline on a suite of synthetic regression tasks.

    This is your notebook's eval_regr_suite, with:
    - imports routed through src/
    - extra MAE metrics added
    """
    y_true: List[float] = []
    y_llm: List[float] = []
    y_gd1: List[float] = []

    for _ in range(n_tasks):
        t = sample_regr_task(d, N, alpha)
        yhat = llm_predict(build_prompt(t))
        if yhat is None:
            continue
        yhat_gd1 = gd1_regr_predict(t.X, t.y, t.x_test, eta)
        y_true.append(t.y_test)
        y_llm.append(yhat)
        y_gd1.append(yhat_gd1)

    if not y_true:
        return {
            "evaluated": 0,
            "mse_llm": float("nan"),
            "mse_gd1": float("nan"),
            "mae_llm": float("nan"),
            "mae_gd1": float("nan"),
            "r2_llm": float("nan"),
            "r2_gd1": float("nan"),
        }

    return {
        "evaluated": len(y_true),
        "mse_llm": mean_squared_error(y_true, y_llm),
        "mse_gd1": mean_squared_error(y_true, y_gd1),
        "mae_llm": mean_absolute_error(y_true, y_llm),
        "mae_gd1": mean_absolute_error(y_true, y_gd1),
        "r2_llm": r2_score(y_true, y_llm),
        "r2_gd1": r2_score(y_true, y_gd1),
    }
