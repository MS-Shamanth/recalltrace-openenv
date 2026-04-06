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

    print("\nGROUND TRUTH:")
    print(env.state()["ground_truth"])

    obs, reward, done, info = env.step(
        {"type": "inspect_node", "node_id": "warehouse"}
    )
    print("\nAFTER INSPECT:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})

    obs, reward, done, info = env.step({"type": "trace_lot", "lot_id": "LotA"})
    print("\nAFTER TRACE:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})

    obs, reward, done, info = env.step(
        {"type": "quarantine", "node_id": "warehouse", "lot_id": "LotA", "quantity": 100}
    )
    print("\nAFTER QUARANTINE:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})

    obs, reward, done, info = env.step({"type": "notify", "node_id": "all"})
    print("\nAFTER NOTIFY:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})

    obs, reward, done, info = env.step({"type": "finalize"})
    print("\nAFTER FINALIZE:")
    print(obs)
    print({"reward": reward, "done": done, "info": info})


if __name__ == "__main__":
    run_smoke_test()
