"""
Probe Training Script for Visual Concept Analysis in Traffic Scenes

This script trains linear or MLP probes on top of pre-extracted features from 
vision-language models to evaluate how well different layers encode specific 
visual concepts in traffic scenes.

The training process:
1. Loads annotations and filters them based on task-specific criteria
2. Splits data into train/val/test sets (by town to ensure generalization)
3. Creates dataloaders for features extracted from multiple model layers
4. Trains probes with learning rate search based on validation
5. Evaluates the best probe on the held-out test set
6. Aggregates results across multiple runs for statistical robustness
"""

import torch
import argparse
import json

from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from utils import (
    get_features_directories,
    dataset_creator,
    dataloader_creator,
    create_probe,
    get_best_results,
    aggregate_best_results,
    save_and_plot_results
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--annotations_path", type=str)           # Path to JSON file with image annotations
parser.add_argument("--parent_features_directory", type=str)  # Directory containing extracted features
parser.add_argument("--num_out", type=int)                    # Number of output classes (1 for binary classification)
parser.add_argument("--distance", type=int, default=None)     # Filter annotations by distance (None = all distances)
parser.add_argument("--epochs", type=int, default=10)         # Number of training epochs per learning rate
parser.add_argument("--num_repeats", type=int, default=10)    # Number of experiment repetitions for statistical significance
parser.add_argument("--use_mlp", action="store_true")         # Use MLP probe instead of linear probe
parser.add_argument("--random_baseline", action="store_true") # Randomize labels to establish baseline performance
parser.add_argument("--save_path", type=str, default="./results")  # Directory to save results
args = parser.parse_args()

# ============================================================================
# LOAD AND PREPROCESS ANNOTATIONS
# ============================================================================
with open(args.annotations_path) as f:
    annotations = json.load(f)

# Special handling for blinker task: remove "no blinker" class (label=0)
# and shift remaining labels to start from 0
if "blinker" in args.annotations_path:
    annotations = [ann for ann in annotations if ann["label"] != 0]
    for ann in annotations:
        ann["label"] -= 1

# Filter annotations by distance if specified
if args.distance:
    selected_annotations = []
    for ann in annotations:
        if ann["distance"] == args.distance or ann["distance"] is None:
            selected_annotations.append(ann)
else:
    selected_annotations = annotations

# ============================================================================
# TRAIN/VALIDATION/TEST SPLIT BY TOWN
# ============================================================================
if "blinker" in args.annotations_path:
    val_town = "Town10HD"
else:
    val_town = "Town12"
if "side" in args.annotations_path:
    test_town = "Town07"
else:
    test_town = "Town15"

# Create test set from designated test town
test_annotations = [ann for ann in selected_annotations if ann["town"] == test_town]

# For "side" task: use stratified random split for train/val (no dedicated val town)
# For other tasks: use dedicated towns for train and validation
if "side" in args.annotations_path:
    trainval_annotations = [ann for ann in selected_annotations if ann["town"] != test_town]
    train_annotations, val_annotations = train_test_split(trainval_annotations, test_size=len(test_annotations) / len(trainval_annotations), random_state=42, stratify=[ann["label"] for ann in trainval_annotations])
else:
    train_annotations = [ann for ann in selected_annotations if (ann["town"] != val_town and ann["town"] != test_town)]
    val_annotations = [ann for ann in selected_annotations if ann["town"] == val_town]
train_size = len(train_annotations)
val_size = len(val_annotations)
test_size = len(test_annotations)

# ============================================================================
# RANDOM BASELINE: Shuffle labels uniformly to measure chance-level performance
# ============================================================================
if args.random_baseline:
    steps = args.num_out if args.num_out > 1 else 2

    # Assign uniform random labels to training set
    cnt = 0
    for i in range(steps):
        for ann in train_annotations[cnt: cnt + train_size // steps]:
            ann["label"] = i
        cnt += train_size // steps

    # Assign uniform random labels to validation set
    cnt = 0
    for i in range(steps):
        for ann in val_annotations[cnt: cnt + val_size // steps]:
            ann["label"] = i
        cnt += val_size // steps

    # Assign uniform random labels to test set
    cnt = 0
    for i in range(steps):
        for ann in test_annotations[cnt: cnt + test_size // steps]:
            ann["label"] = i
        cnt += test_size // steps

# ============================================================================
# FEATURE DIRECTORY SETUP
# Configure paths to pre-extracted features from different model layers
# ============================================================================
features_directories = get_features_directories(args.parent_features_directory)

# ============================================================================
# DATASET AND DATALOADER CREATION
# ============================================================================
train_datasets = dataset_creator(train_annotations, features_directories)
val_datasets = dataset_creator(val_annotations, features_directories)
test_datasets = dataset_creator(test_annotations, features_directories)

train_dataloaders = dataloader_creator(train_datasets, batch_size=32, shuffle=True)
val_dataloaders = dataloader_creator(val_datasets, batch_size=32, shuffle=False)
test_dataloaders = dataloader_creator(test_datasets, batch_size=32, shuffle=False)

# Learning rates to search over for hyperparameter optimization
learning_rates = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
all_best_results = []
all_best_probes = defaultdict(list)

if args.num_out == 1:
    loss_fn = torch.nn.BCEWithLogitsLoss()
else:
    loss_fn = torch.nn.CrossEntropyLoss()

for _ in tqdm(range(args.num_repeats)):
    # Nested dict: results[layer][learning_rate][metric] = list of values
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Train a separate probe for each model layer
    for level in tqdm(train_datasets.keys()):
        train_loader, val_loader, test_loader = train_dataloaders[level], val_dataloaders[level], test_dataloaders[level]
        
        # Track best probe across all learning rates and epochs
        best_probe = None
        max_val_acc = -1
        best_lr = None
        best_epoch = None
        
        # Grid search over learning rates
        for lr in learning_rates:
            input_size = train_datasets[level][0][0].shape[0]
            probe = create_probe(input_size, args.num_out, args.use_mlp)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
            
            for epoch in range(args.epochs):
                # training
                probe.train()
                overall_train_loss = 0.0
                overall_train_acc = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    if args.num_out == 1:
                        labels = labels.to(torch.float32).unsqueeze(-1)

                    outputs = probe(inputs)
                    train_loss = loss_fn(outputs, labels)
                    overall_train_loss += inputs.shape[0] * train_loss.item()

                    if args.num_out == 1:
                        outputs = torch.sigmoid(outputs)
                        outputs = outputs.squeeze()
                        labels = labels.squeeze()
                        train_acc = (outputs.round() == labels).float().mean()
                    else:
                        train_acc = (outputs.argmax(dim=1) == labels).float().mean()
                    overall_train_acc += inputs.shape[0] * train_acc.item()

                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                # Record epoch training metrics
                results[level][lr]["train_losses"].append(overall_train_loss / train_size)
                results[level][lr]["train_accuracies"].append(overall_train_acc / train_size)

                # validation
                probe.eval()
                overall_val_loss = 0.0
                overall_val_acc = 0.0
                
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    if args.num_out == 1:
                        labels = labels.to(torch.float32).unsqueeze(-1)
                    
                    with torch.inference_mode():
                        outputs = probe(inputs)
                    val_loss = loss_fn(outputs, labels)
                    overall_val_loss += inputs.shape[0] * val_loss.item()

                    if args.num_out == 1:
                        outputs = torch.sigmoid(outputs)
                        outputs = outputs.squeeze()
                        labels = labels.squeeze()
                        val_acc = (outputs.round() == labels).float().mean()
                    else:
                        val_acc = (outputs.argmax(dim=1) == labels).float().mean()
                    overall_val_acc += inputs.shape[0] * val_acc.item()

                # Record epoch validation metrics
                results[level][lr]["val_losses"].append(overall_val_loss / val_size)
                results[level][lr]["val_accuracies"].append(overall_val_acc / val_size)
                
                # Track the best model based on validation accuracy
                current_val_acc = overall_val_acc / val_size
                if current_val_acc >= max_val_acc:
                    max_val_acc = current_val_acc
                    best_probe = deepcopy(probe.state_dict())
                    best_lr = lr
                    best_epoch = epoch

        # test
        all_best_probes[level].append(best_probe)

        # Reload the best probe weights for testing
        input_size = train_datasets[level][0][0].shape[0]
        probe = create_probe(input_size, args.num_out, args.use_mlp)
        probe.load_state_dict(best_probe)
        probe.eval()
        
        overall_test_loss = 0.0
        overall_test_acc = 0.0
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            if args.num_out == 1:
                labels = labels.to(torch.float32).unsqueeze(-1)

            with torch.inference_mode():
                outputs = probe(inputs) 
            test_loss = loss_fn(outputs, labels)
            overall_test_loss += inputs.shape[0] * test_loss.item()

            if args.num_out == 1:
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze()
                labels = labels.squeeze()
                test_acc = (outputs.round() == labels).float().mean()
            else:
                test_acc = (outputs.argmax(dim=1) == labels).float().mean()
            overall_test_acc += inputs.shape[0] * test_acc.item()

        # Store test results with the best learning rate and epoch info
        results[level][best_lr]["test_loss"] = (overall_test_loss / test_size, best_epoch)
        results[level][best_lr]["test_accuracy"] = (overall_test_acc / test_size, best_epoch)

    # Extract best results for this repetition
    best_results = get_best_results(results)
    all_best_results.append(best_results)

# Combine results across all repetitions
aggregated_results = aggregate_best_results(all_best_results)

category = args.annotations_path.split("/")[-2]

# Save results, generate plots, and store best probe weights
save_and_plot_results(aggregated_results, args, list(train_datasets.keys()), all_best_probes, category)
