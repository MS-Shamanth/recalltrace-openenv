from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.env import RecallTraceEnv
from scenario.scenario import get_scenario
from grader.grader import grade, compute_reward


def run_smoke_test():
    # test all levels
    for level in ["easy", "medium", "hard"]:
        print(f"\n===== TESTING LEVEL: {level.upper()} =====")

        scenario = get_scenario(level)
        env = RecallTraceEnv(scenario=scenario)

        obs = env.reset()
        print("\nRESET OBSERVATION:")
        print(obs)

        # simulate agent inspecting warehouse
        obs, reward, done, info = env.step(
            {"type": "inspect_node", "node_id": "warehouse"}
        )

        print("\nAFTER INSPECT:")
        print(obs)

        # -------------------------
        # 🔥 SIMULATED AGENT OUTPUT
        # -------------------------
        agent_output = {
            "quarantine": []
        }

        for node, lots in scenario["inventory"].items():
            if scenario["contaminated_lot"] in lots:
                agent_output["quarantine"].append({
                    "node": node,
                    "lot": scenario["contaminated_lot"],
                    "qty": lots[scenario["contaminated_lot"]]
                })

        # -------------------------
        # 🧠 GRADING + REWARD
        # -------------------------
        score = grade(agent_output, scenario)
        reward = compute_reward(agent_output, scenario)

        print("\nAGENT OUTPUT:")
        print(agent_output)

        print("\nSCORE:", score)
        print("REWARD:", reward)

        # finalize env
        obs, reward, done, info = env.step({"type": "finalize"})

        print("\nAFTER FINALIZE:")
        print(obs)
        print({"reward": reward, "done": done, "info": info})


if __name__ == "__main__":
    run_smoke_test()