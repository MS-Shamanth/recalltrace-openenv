import os
import json
import time
import requests
from huggingface_hub import InferenceClient

# Configuration
SPACE_URL = "https://ms-shamanth-recalltrace-openenv.hf.space"
# Local Model Configuration
MODEL_ID = "ms-shamanth/recalltrace-investigator"

def get_system_prompt():
    return """You are a supply-chain investigator AI.
You must identify and contain contamination in a supply chain graph.

AVAILABLE TOOLS:
1. {"type": "inspect_node", "node_id": "W1"} -> Inspect a node's inventory and evidence.
2. {"type": "trace_lot", "lot_id": "L123"} -> Trace a specific lot through the supply chain.
3. {"type": "cross_reference", "node_id": "W1"} -> Compare physical inventory with records.
4. {"type": "request_lab_test", "node_id": "W1"} -> Run a 24-hour pathogen test on a node.
5. {"type": "quarantine", "node_id": "W1", "rationale": "reason"} -> Quarantine a node.
6. {"type": "notify", "node_id": "W1"} -> Alert downstream nodes.
7. {"type": "finalize", "rationale": "reason"} -> Submit final containment strategy.

OUTPUT FORMAT:
You must output a SINGLE valid JSON object matching one of the tools above. Do not output any markdown formatting, thoughts, or extra text. Just the JSON object.
"""

def format_observation(obs: dict) -> str:
    """Format the environment state into a text prompt."""
    prompt = "CURRENT OBSERVATION:\n"
    
    budget = obs.get("budget", {})
    prompt += f"Steps Remaining: {budget.get('steps_remaining', '?')}\n\n"
    
    state = obs.get("state", {})
    prompt += "KNOWLEDGE BASE:\n"
    
    nodes = state.get("nodes", {})
    if not nodes:
        prompt += "No nodes investigated yet.\n"
    else:
        for nid, ndata in nodes.items():
            prompt += f"- Node {nid}:\n"
            if ndata.get("inventory"):
                prompt += f"  Inventory: {', '.join(ndata['inventory'])}\n"
            if ndata.get("evidence_strength") is not None:
                prompt += f"  Contamination Evidence: {ndata['evidence_strength']:.2f}\n"
            if ndata.get("quarantined_inventory"):
                prompt += f"  Status: QUARANTINED\n"
                
    if obs.get("feedback"):
        prompt += f"\nPREVIOUS ACTION FEEDBACK:\n{obs['feedback']}\n"
        
    prompt += "\nDecide your next action and output ONLY valid JSON:"
    return prompt

def run_loop():
    print(f"--- Starting Live Inference Loop ---")
    print(f"Environment: {SPACE_URL}")
    print(f"Model: {MODEL_ID} (Running locally via Unsloth)")
    
    # 1. Initialize Local Model via Unsloth
    print("\nLoading local model into GPU memory...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("Model loaded successfully!")
    except ImportError:
        print("Error: 'unsloth' is not installed. Please run this script in the environment where you trained the model.")
        return
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
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
    
    messages = [
        {"role": "system", "content": get_system_prompt()}
    ]
    
    while not done and step_count < 25:
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        
        obs_text = format_observation(obs)
        messages.append({"role": "user", "content": obs_text})
        
        print("Running local GPU inference...")
        try:
            # Format using tokenizer chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            outputs = model.generate(input_ids=inputs, max_new_tokens=150, use_cache=True, temperature=0.1)
            raw_action = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            
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
            fallback_node = "W1"
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
            
            obs = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0)
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
    
    # In a real HF Job, we could push these metrics to a Dataset.
    # For now, we print them out.
    print("\nTo log this to a Hugging Face Dataset automatically, you can add code here to use:")
    print("from huggingface_hub import Repository")
    print("or use datasets.Dataset.push_to_hub()")

if __name__ == "__main__":
    run_loop()
