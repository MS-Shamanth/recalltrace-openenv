import os
import json
import time
import requests
from huggingface_hub import InferenceClient

# Configuration
SPACE_URL = "https://ms-shamanth-recalltrace-openenv.hf.space"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Get HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

def get_system_prompt():
    return """You are a supply-chain investigator AI.
You must identify and contain contamination in a supply chain graph.

AVAILABLE TOOLS:
1. {"type": "inspect_node", "node_id": "warehouse_0"} -> Inspect a node's inventory and evidence.
2. {"type": "trace_lot", "lot_id": "LotA"} -> Trace a specific lot through the supply chain.
3. {"type": "cross_reference", "node_id": "warehouse_0"} -> Compare physical inventory with records.
4. {"type": "request_lab_test", "node_id": "warehouse_0"} -> Run a 24-hour pathogen test on a node.
5. {"type": "quarantine", "node_id": "warehouse_0", "lot_id": "LotA", "rationale": "reason"} -> Quarantine a specific lot at a node.
6. {"type": "notify", "node_id": "warehouse_0"} -> Alert downstream nodes.
7. {"type": "finalize", "rationale": "reason"} -> Submit final containment strategy.

OUTPUT FORMAT:
You must output a SINGLE valid JSON object matching one of the tools above. Do not output any markdown formatting, thoughts, or extra text. Just the JSON object. You must ONLY use nodes and lots explicitly present in the CURRENT OBSERVATION.
"""

def format_observation(obs: dict) -> str:
    """Format the environment state into a text prompt."""
    prompt = "CURRENT OBSERVATION:\n"
    
    steps = obs.get("remaining_step_budget", "?")
    prompt += f"Steps Remaining: {steps}\n\n"
    
    notice = obs.get("recall_notice", "")
    if notice:
        prompt += f"RECALL NOTICE: {notice}\n\n"
        
    prompt += "KNOWLEDGE BASE:\n"
    
    inventory = obs.get("inventory", {})
    belief = obs.get("belief_state", {})
    inspected = obs.get("inspected_nodes", [])
    quarantined = obs.get("quarantined_inventory", {})
    
    all_nodes = set(list(inventory.keys()) + list(belief.keys()))
    if not all_nodes:
        prompt += "No nodes investigated yet.\n"
    else:
        for nid in sorted(list(all_nodes)):
            prompt += f"- Node {nid}:\n"
            if nid in inventory:
                inv_str = ", ".join([f"{k}: {v}" for k,v in inventory[nid].items()])
                prompt += f"  Inventory: {inv_str}\n"
            if nid in belief:
                prompt += f"  Suspicion Score: {belief[nid]:.2f}\n"
            if nid in inspected:
                prompt += f"  Status: INSPECTED\n"
            if nid in quarantined:
                prompt += f"  Status: QUARANTINED\n"
                
    history = obs.get("history", [])
    if history:
        prompt += f"\nPREVIOUS ACTION FEEDBACK:\n{history[-1]}\n"
        
    prompt += "\nDecide your next action and output ONLY valid JSON:"
    return prompt

def run_loop():
    print(f"--- Starting Live API Inference Loop ---")
    print(f"Environment: {SPACE_URL}")
    print(f"Model: {MODEL_ID} (via HF Serverless API)")
    
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN environment variable is not set. Inference API may rate-limit or reject the request.")
    
    # 1. Initialize API Client
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    
    # 2. Reset Environment
    print("\n[1] Resetting remote environment...")
    try:
        res = requests.post(f"{SPACE_URL}/reset", json={"num_nodes": 10}, timeout=15)
        res.raise_for_status()
        obs = res.json()
        print("Environment reset successful.")
    except Exception as e:
        print(f"Failed to connect to Space: {e}")
        return

    # 3. Main Loop
    done = False
    step_count = 0
    total_reward = 0.0
    action_history = []
    
    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]
    
    while not done and step_count < 25:
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        
        obs_text = format_observation(obs)
        messages.append({"role": "user", "content": obs_text})
        
        print("Querying Hugging Face Inference API...")
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                top_p=0.9
            )
            raw_action = response.choices[0].message.content.strip()
            
            if raw_action.startswith("```json"):
                raw_action = raw_action[7:-3].strip()
            elif raw_action.startswith("```"):
                raw_action = raw_action[3:-3].strip()
                
            action_dict = json.loads(raw_action)
            print(f"Agent chose action: {action_dict}")
            
            messages.append({"role": "assistant", "content": raw_action})
            
        except Exception as e:
            print(f"Model Inference Failed: {e}")
            print("Fallback to safe action: inspect_node on first available node")
            fallback_node = "warehouse_0"
            if obs.get("inventory"):
                fallback_node = list(obs["inventory"].keys())[0]
            elif obs.get("belief_state"):
                fallback_node = list(obs["belief_state"].keys())[0]
            action_dict = {"type": "inspect_node", "node_id": fallback_node}
            
        # Step Environment
        print("Sending action to remote Space...")
        try:
            res = requests.post(f"{SPACE_URL}/step", json=action_dict, timeout=15)
            res.raise_for_status()
            step_data = res.json()
            
            reward = step_data.get("reward", 0.0)
            
            action_history.append({
                "obs": obs,
                "action": action_dict,
                "reward": reward
            })
            
            obs = step_data.get("observation", {})
            done = step_data.get("done", False)
            info = step_data.get("info", {})
            
            total_reward += reward
            print(f"Reward: {reward:.4f} | Done: {done}")
            
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Failed to step environment: {e} - {e.response.text}")
            else:
                print(f"Failed to step environment: {e}")
            break

    print("\n--- Episode Complete ---")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.4f}")
    
    try:
        from datasets import Dataset
        from huggingface_hub import whoami
        import uuid
        import pandas as pd
        
        print("\nSaving trace to Hugging Face Dataset...")
        ep_id = str(uuid.uuid4())
        
        data = {
            "episode_id": [ep_id] * len(action_history),
            "step": list(range(len(action_history))),
            "observation": [json.dumps(h["obs"]) for h in action_history],
            "action": [json.dumps(h["action"]) for h in action_history],
            "reward": [h["reward"] for h in action_history]
        }
        
        ds = Dataset.from_dict(data)
        
        # Get username from token to push to the correct namespace
        username = whoami(token=HF_TOKEN, cache=True)["name"]
        dataset_repo = f"{username}/recalltrace-inference-logs"
        
        ds.push_to_hub(dataset_repo, token=HF_TOKEN, private=True)
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{dataset_repo}")
    except ImportError:
        print("\n'datasets' package not found. Skipping Hugging Face dataset upload.")
        print("To log this to a Hugging Face Dataset automatically, install the `datasets` package:")
        print("pip install datasets pandas")
    except Exception as e:
        print(f"\nFailed to push dataset to Hugging Face: {e}")

if __name__ == "__main__":
    run_loop()
