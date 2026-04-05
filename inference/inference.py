"""Quick smoke test for the Phase 1 RecallTrace environment."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.env import RecallTraceEnv


def run_smoke_test():
    env = RecallTraceEnv()

    obs = env.reset()
    print("RESET OBSERVATION:")
    print(obs)

    obs, reward, done, info = env.step(
        {"type": "inspect_node", "node_id": "warehouse"}
    )
    print("\nAFTER INSPECT:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})

    obs, reward, done, info = env.step({"type": "finalize"})
    print("\nAFTER FINALIZE:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})


if __name__ == "__main__":
    run_smoke_test()
