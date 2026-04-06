# grader/grader.py

def compute_ground_truth(scenario):
    contaminated = scenario["contaminated_lot"]
    inventory = scenario["inventory"]
    transformations = scenario.get("transformations", {})

    # handle relabeling
    affected_lot = transformations.get(contaminated, contaminated)

    ground_truth = {}

    for node, lots in inventory.items():
        if affected_lot in lots:
            ground_truth[node] = {
                "lot": affected_lot,
                "qty": lots[affected_lot]
            }

    return ground_truth


# ---------------- GRADING ----------------
def grade(agent_output, scenario):
    """
    agent_output:
    {
        "quarantine": [
            {"node": "store1", "lot": "LotA", "qty": 50}
        ]
    }
    """

    ground_truth = compute_ground_truth(scenario)

    score = 0
    total = len(ground_truth)

    for action in agent_output.get("quarantine", []):
        node = action["node"]
        lot = action["lot"]
        qty = action["qty"]

        if node in ground_truth:
            gt = ground_truth[node]

            if lot == gt["lot"]:
                score += 0.5

                if qty == gt["qty"]:
                    score += 0.5

    return score / total if total > 0 else 0


# ---------------- REWARD ----------------
def compute_reward(agent_output, scenario):
    ground_truth = compute_ground_truth(scenario)

    reward = 0

    for action in agent_output.get("quarantine", []):
        node = action["node"]
        lot = action["lot"]
        qty = action["qty"]

        if node in ground_truth:
            gt = ground_truth[node]

            if lot == gt["lot"]:
                reward += 10
            else:
                reward -= 5

            if qty == gt["qty"]:
                reward += 10
            else:
                reward -= 3
        else:
            reward -= 5

    # bonus
    if grade(agent_output, scenario) == 1.0:
        reward += 20

    return reward