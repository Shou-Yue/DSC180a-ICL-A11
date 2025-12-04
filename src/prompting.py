import re

from .tasks import RegrTask

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
TOKEN_RE = re.compile(r"(?<!\d)[\+\-]?1(?!\d)")  # standalone -1, +1, or 1


def fmtv(x, digits: int = 5) -> str:
    return ", ".join(f"{v:.{digits}f}" for v in x.tolist())


def build_prompt(t: RegrTask, digits: int = 5) -> str:
    ctx = "\n".join(
        [
            f"({fmtv(t.X[i], digits)}) -> {t.y[i]:.{digits}f}"
            for i in range(t.X.shape[0])
        ]
    )
    xs = fmtv(t.x_test, digits)
    return (
        "You are solving a regression task \n"
        "Given training pairs (x -> y), predict unseen y"
        f"{ctx}\n\nx_test = [{xs}]\n"
        "Respond with a single float rounded to 4 digits, ex: 1.5344"
    )
