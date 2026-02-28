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
    get_save_path,
    get_tile_grid_info,
)


# ----------------------------------------------------------------------------
# left/right split selection for region pooling
#
# When no numeric split is specified for a (model, category, distance) tuple,
# we use None (which defaults to width//2 in the pooling code).
# ----------------------------------------------------------------------------

_AUTO_SPLITS = {
    "ovis2.5": {
        # category 2: pedestrian_direction
        2: {5: 48, 10: 54, 20: 57, 30: 58, 40: 58, 50: 59},
        # category 8: bicycle_direction
        8: {5: 48, 10: 54, 20: 57, 30: 58, 40: 58, 50: 59},
    },
    "internvl3.5": {
        2: {5: 52, 10: 58, 20: 61, 30: 62, 40: 62, 50: 63},
        8: {5: 52, 10: 58, 20: 61, 30: 62, 40: 62, 50: 63},
    },
    "vst": {
        2: {5: 38, 10: 42, 20: 45, 30: 45, 40: 46, 50: 46},
        8: {5: 38, 10: 42, 20: 45, 30: 45, 40: 46, 50: 46},
    },
}


def _infer_split(model_name: str, category: int, distance: int | None) -> int | None:
    if not distance:
        return None
    return _AUTO_SPLITS.get(model_name, {}).get(int(category), {}).get(int(distance))

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, required=True, choices=["ovis2.5", "internvl3.5", "vst"],
                       help="Model to use for feature extraction")
argparser.add_argument("--annotations_path", type=str, required=True)
argparser.add_argument("--llm_features", type=str, default="visual_embs_and_last_token",
                       choices=["visual_embs", "last_token", "all", "visual_embs_and_last_token"])
argparser.add_argument("--category", type=int, default=0)
argparser.add_argument("--save_path", type=str, default="./extracted_features")
# Region pooling arguments
argparser.add_argument("--distance", type=int, default=0)
argparser.add_argument("--region_pooling", dest="region_pooling", action="store_true",
                       help="Pool features separately over left/right halves of the visual token grid")
# Model-specific arguments
argparser.add_argument("--use_cls", action="store_true", help="InternVL3.5: whether to use CLS token")
argparser.add_argument("--num_tiles", type=int, default=9, help="InternVL3.5: number of tiles")
argparser.add_argument("--version", type=str, default="rl", choices=["rl", "sft"], help="VST: model version")
args = argparser.parse_args()

# Determine split for region pooling.
args.split = _infer_split(args.model, args.category, getattr(args, "distance", None))

# Load annotations
with open(args.annotations_path) as f:
    annotations = json.load(f)

if "blinker" in args.annotations_path:
    annotations = [ann for ann in annotations if ann["label"] != 0]

if args.distance:
    annotations = [ann for ann in annotations if ann["distance"] == args.distance]

if len(annotations) == 0:
    raise ValueError(
        "No annotations left after filtering. "
        "Check --annotations_path and (if used) --distance."
    )

# Get query
question = get_query(args.category)

# Load model and processor/tokenizer
model, processor = load_model(args.model, args)

# Prepare initial inputs to determine indices for hooks
initial_inputs = preprocess_inputs(args.model, model, processor, annotations[0], question, args)

# Region pooling state (mirrors probing_research scripts)
grid_info = {} if args.region_pooling else None
window_index = None
indices_ref = {"indices": None} if args.model == "vst" else None

if args.model == "ovis2.5" and args.region_pooling:
    window_index, _ = model.visual_tokenizer.vit.vision_model.encoder.get_window_index(initial_inputs["grid_thws"])
elif args.model == "vst" and args.region_pooling:
    window_index = model.model.visual.get_window_index(initial_inputs["image_grid_thw"])[0]

# Pass pooling context into save-hook factory (kept on args to avoid widening APIs)
args._grid_info = grid_info
args._window_index = window_index

# Initialize feature storage
average_features = defaultdict(list)

# Create save hook
save_hook = get_save_hook(args.model, args, average_features)

# Register hooks for all model components
hook_state = {"indices_ref": indices_ref} if args.model == "vst" else None
register_hooks(args.model, model, save_hook, initial_inputs, hook_state=hook_state)

# Extract features for all annotations
for ann in tqdm(annotations):
    inputs = preprocess_inputs(args.model, model, processor, ann, question, args)

    # Update per-image state for region pooling / VST indices
    if args.model == "ovis2.5" and args.region_pooling and inputs.get("grid_thws", None) is not None:
        _, gh, gw = inputs["grid_thws"][0].tolist()
        gh, gw = int(gh), int(gw)
        grid_info["vit_h"] = gh
        grid_info["vit_w"] = gw
        grid_info["merged_h"] = gh // 2
        grid_info["merged_w"] = gw // 2
    elif args.model == "internvl3.5" and args.region_pooling:
        grid_info.update(get_tile_grid_info(ann["image_path"], max_num=12))
    elif args.model == "vst":
        # Update indices for this image (matches legacy `vst.py` behavior)
        if indices_ref is not None:
            new_indices = (inputs.input_ids[0] == 151655).nonzero(as_tuple=True)[0]
            indices_ref["indices"] = new_indices

        if args.region_pooling:
            image_grid_thw = inputs["image_grid_thw"]
            _, h, w = image_grid_thw[0].tolist()
            h, w = int(h), int(w)
            grid_info.update({
                "vit_h": h,
                "vit_w": w,
                "merged_h": h // 2,
                "merged_w": w // 2,
            })
    
    with torch.inference_mode():
        run_inference(args.model, model, inputs)

# Save features
save_path = get_save_path(args.model, args)
if args.region_pooling:
    save_path = save_path.replace("features", "rp_features")

for key, value in average_features.items():
    for feature, ann in zip(value, annotations):
        name = ann["image_path"].split("/")[-1][:-4]
        sample_save_path = f"{save_path}/{key}"
        os.makedirs(sample_save_path, exist_ok=True)
        torch.save(feature.to(torch.float32).cpu(), f"{sample_save_path}/{name}.pt")

print(f"Features saved to {save_path}")
