import torch
import argparse
import json
import os

from utils import (
    load_model, 
    get_tokenizer, 
    get_answer_ids, 
    prepare_inputs, 
    forward_pass, 
    get_query, 
    get_answer
)
from collections import defaultdict
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, required=True, choices=["ovis2.5", "internvl3.5", "vst-rl", "vst-sft"])
argparser.add_argument("--annotations_path", type=str, required=True)
argparser.add_argument("--category", type=int, required=True)
argparser.add_argument("--answers", nargs="+", type=str, required=True)
argparser.add_argument("--save_path", type=str, default="./results")
argparser.add_argument("--question_only", action="store_true")
args = argparser.parse_args()

if args.question_only:
    save_path = args.save_path + f"/question_only/{args.model}_category_{args.category}"
else:
    save_path = args.save_path + f"/{args.model}_category_{args.category}"
os.makedirs(save_path, exist_ok=True)

with open(args.annotations_path) as f:
    annotations = json.load(f)

test_town = "Town15" if "side" not in args.annotations_path else "Town07"
annotations = [ann for ann in annotations if ann["town"] == test_town]

if args.category == 4:
    annotations = [ann for ann in annotations if ann["label"] != 0]

    for ann in annotations:
        ann["label"] -= 1

# Load model and tokenizer
model = load_model(args.model)
tokenizer = get_tokenizer(args.model, model)

# Get answer token IDs
answer_ids = get_answer_ids(tokenizer, args.answers)

# Get question
question = get_query(args.category, args.question_only)

# Prepare static inputs if needed
static_inputs = prepare_inputs(args.model, model, tokenizer, question, image_path=None, static_only=True)

detailed_results = defaultdict(list)
correct_samples = []
incorrect_samples = []

for ann in tqdm(annotations):
    # Prepare inputs for this specific image
    inputs = prepare_inputs(args.model, model, tokenizer, question, image_path=ann["image_path"], static_inputs=static_inputs)
    
    gt_answer = get_answer(args.category, ann["label"])

    # Forward pass
    with torch.inference_mode():
        outputs = forward_pass(args.model, model, inputs)

    # Get next token logits and predict
    next_token_logits = outputs.logits[:, -1, :]
    
    if args.question_only:
        # Compare probabilities of answer tokens only
        answer_logits = [next_token_logits[0, idx] for idx in answer_ids]
        pred_id = torch.argmax(torch.tensor(answer_logits)).item()
        predicted_token = args.answers[pred_id]
    else:
        # Use the token with highest probability across the whole vocabulary
        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
        actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        predicted_token = actual_tokenizer.decode([predicted_token_id])

    if predicted_token.strip().lower() == gt_answer.strip().lower():
        detailed_results[ann["distance"]].append(1)
        detailed_results["overall"].append(1)
        correct_samples.append(ann)
    else:
        detailed_results[ann["distance"]].append(0)
        detailed_results["overall"].append(0)
        incorrect_samples.append(ann)

percentage_results = {key: sum(value) / len(value) for key, value in detailed_results.items()}

if None in percentage_results:
    percentage_results_probe_style = {key: (val + percentage_results[None]) / 2 for key, val in percentage_results.items() if key not in [None, "overall"]}

with open(f"{save_path}/detailed_results.json", "w") as f:
    json.dump(detailed_results, f)

with open(f"{save_path}/percentage_results.json", "w") as f:
    json.dump(percentage_results, f)

if None in percentage_results:
    with open(f"{save_path}/percentage_results_probe_style.json", "w") as f:
        json.dump(percentage_results_probe_style, f)

with open(f"{save_path}/correct_samples.json", "w") as f:
    json.dump(correct_samples, f)

with open(f"{save_path}/incorrect_samples.json", "w") as f:
    json.dump(incorrect_samples, f)
