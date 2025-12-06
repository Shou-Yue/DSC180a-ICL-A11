import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------
# 1. Project root + sys.path
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------
# 2. Load .env from project root
# ---------------------------------------------------
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

print("OPENAI_API_KEY visible:", bool(os.getenv("OPENAI_API_KEY")))

# ---------------------------------------------------
# 3. Import the correct function
# ---------------------------------------------------
from src.results import run_ood_scaling


def main() -> None:
    results_dir = PROJECT_ROOT / "results"

    d = 40
    N = 2 * d + 1
    eta = 1.0
    alphas = [0.5, 1.0, 1.5, 2.0]

    results = run_ood_scaling(
        alphas=alphas,
        d=d,
        N=N,
        eta=eta,
        n_tasks=5,
        results_dir=str(results_dir),
    )

    print("Done. Results saved to", results_dir)
    print(results)


if __name__ == "__main__":
    main()
