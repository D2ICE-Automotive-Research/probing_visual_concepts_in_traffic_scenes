import torch
import argparse
import json
import os

from collections import defaultdict
from tqdm import tqdm
from utils import (
    get_query,
    load_model,
    get_save_hook,
    register_hooks,
    preprocess_inputs,
    run_inference,
    get_save_path
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, required=True, choices=["ovis2.5", "internvl3.5", "vst"],
                       help="Model to use for feature extraction")
argparser.add_argument("--annotations_path", type=str, required=True)
argparser.add_argument("--llm_features", type=str, default="visual_embs_and_last_token",
                       choices=["visual_embs", "last_token", "all", "visual_embs_and_last_token"])
argparser.add_argument("--category", type=int, default=0)
argparser.add_argument("--save_path", type=str, default="./extracted_features")
# Model-specific arguments
argparser.add_argument("--use_cls", action="store_true", help="InternVL3.5: whether to use CLS token")
argparser.add_argument("--num_tiles", type=int, default=9, help="InternVL3.5: number of tiles")
argparser.add_argument("--version", type=str, default="rl", choices=["rl", "sft"], help="VST: model version")
args = argparser.parse_args()

# Load annotations
with open(args.annotations_path) as f:
    annotations = json.load(f)

if "blinker" in args.annotations_path:
    annotations = [ann for ann in annotations if ann["label"] != 0]

# Get query
question = get_query(args.category)

# Load model and processor/tokenizer
model, processor = load_model(args.model, args)

# Prepare initial inputs to determine indices for hooks
initial_inputs = preprocess_inputs(args.model, model, processor, annotations[0], question, args)

# Initialize feature storage
average_features = defaultdict(list)

# Create save hook
save_hook = get_save_hook(args.model, args, average_features)

# Register hooks for all model components
register_hooks(args.model, model, save_hook, initial_inputs)

# Extract features for all annotations
for ann in tqdm(annotations[:10]):
    inputs = preprocess_inputs(args.model, model, processor, ann, question, args)
    
    with torch.inference_mode():
        run_inference(args.model, model, inputs)

# Save features
save_path = get_save_path(args.model, args)

for key, value in average_features.items():
    for feature, ann in zip(value, annotations):
        name = ann["image_path"].split("/")[-1][:-4]
        sample_save_path = f"{save_path}/{key}"
        os.makedirs(sample_save_path, exist_ok=True)
        torch.save(feature.to(torch.float32).cpu(), f"{sample_save_path}/{name}.pt")

print(f"Features saved to {save_path}")
