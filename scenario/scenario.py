# scenario/scenario.py

"""
Scenario definitions for RecallTrace Phase 1
"""

def get_scenario(level="easy"):
    if level == "easy":
        return easy_scenario()
    elif level == "medium":
        return medium_scenario()
    elif level == "hard":
        return hard_scenario()
    else:
        raise ValueError("Invalid level")


# ---------------- EASY ----------------
def easy_scenario():
    return {
        "shipment_path": ["warehouse", "store1", "store2"],
        "inventory": {
            "warehouse": {"LotA": 100},
            "store1": {"LotA": 100},
            "store2": {"LotA": 100}
        },
        "contaminated_lot": "LotA",
        "recall_notice": "Lot A is contaminated",
        "transformations": {}
    }


# ---------------- MEDIUM ----------------
def medium_scenario():
    return {
        "shipment_path": ["warehouse", "store1", "store2"],
        "inventory": {
            "warehouse": {"LotA": 100},
            "store1": {"LotA1": 100},
            "store2": {"LotA1": 100}
        },
        "contaminated_lot": "LotA",
        "recall_notice": "Lot A is contaminated",
        "transformations": {
            "LotA": "LotA1"
        }
    }


# ---------------- HARD ----------------
def hard_scenario():
    return {
        "shipment_path": ["warehouse", "store1", "store2"],
        "inventory": {
            "warehouse": {"LotA": 100},
            "store1": {"LotA": 50, "LotB": 50},
            "store2": {"LotA": 50, "LotB": 50}
        },
        "contaminated_lot": "LotA",
        "recall_notice": "Lot A is contaminated",
        "transformations": {}
    }