"""
Utility functions for training linear/MLP probes on visual language model features.

This module provides:
- Feature directory discovery and ordering
- Dataset and DataLoader creation for probe training
- Probe model creation (linear and MLP variants)
- Results aggregation and visualization utilities
"""

import torch
import json
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


# =============================================================================
# Feature Directory Utilities
# =============================================================================

def get_features_directories(parent_directory):
    """
    Discover and order all feature directories within a parent directory.
    
    Searches recursively for directories containing .pt (PyTorch tensor) files,
    excluding MLP and self-attention subdirectories. Returns directories ordered
    by model component (vision_encoder -> projector -> language_model) and layer number.
    """
    # Step 1: Find all directories containing .pt feature files
    features_directories = []
    for dirpath, _, filenames in os.walk(parent_directory):
        if filenames:
            if filenames[0].endswith(".pt"):
                features_directories.append(dirpath)

    # Step 2: Extract readable directory names for sorting
    directory_names = []
    for features_directory in features_directories:
        if features_directory.endswith("projector"):
            directory_names.append("projector")
        else:
            # Extract component/layer format, e.g., "vision_encoder/layer_0"
            directory_names.append("/".join(features_directory.split("/")[-2:]))

    # Step 3: Sort directories by component order
    # Order: vision_encoder layers -> projector -> language_model layers
    components_order = ["vision_encoder", "projector", "language_model"]
    projector_included = False
    if "projector" in directory_names:
        directory_names.remove("projector")
        projector_included = True
    
    # First sort by layer number (within each component)
    directory_names.sort(key=lambda x: int(x.split("layer_")[-1]) if ("layer_" in x and x.split("layer_")[-1].isdigit()) else 100)
    # Then sort by component order (stable sort preserves layer ordering)
    directory_names.sort(key=lambda x: components_order.index(x.split("/")[0]))
    
    # Insert projector after all vision encoder layers
    for i, name in enumerate(directory_names[::-1]):
        if "vision_encoder" in name:
            break
    if projector_included:
        directory_names.insert(len(directory_names) - i, "projector")

    # Step 4: Map sorted names back to full directory paths
    ordered_features_directories = []   
    for name in directory_names:
        for d in features_directories:
            if d.endswith(name):
                ordered_features_directories.append(d)
                break

    return ordered_features_directories


# =============================================================================
# Dataset and DataLoader Classes
# =============================================================================

class ProbeDataset(Dataset):
    """
    PyTorch Dataset for loading pre-extracted features and their labels.
    
    Each sample consists of a feature tensor (loaded from a .pt file) and
    its corresponding label from the annotations.
    """
    def __init__(self, annotations, features_directory):
        self.annotations = annotations
        self.features_directory = features_directory

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Construct feature file path by replacing image extension with .pt
        feature_path = os.path.join(self.features_directory, self.annotations[idx]["image_path"].split("/")[-1].replace(".jpg", ".pt"))
        feature = torch.load(feature_path)
        label = self.annotations[idx]["label"]

        return feature, label


def dataset_creator(annotations, features_directories):
    """
    Create a dictionary of ProbeDatasets, one for each feature directory.
    """
    datasets = {}
    for features_directory in features_directories:
        dataset = ProbeDataset(annotations, features_directory)
        # Use simplified directory name as key
        if features_directory.endswith("projector"):
            directory_name = "projector"
        else:
            directory_name = "/".join(features_directory.split("/")[-2:])
        datasets[directory_name] = dataset

    return datasets


def dataloader_creator(datasets, batch_size, shuffle):
    """
    Create DataLoaders for each dataset in the dictionary.
    """
    dataloaders = {}
    for directory_name, dataset_list in datasets.items():
        dataloader = DataLoader(dataset_list, batch_size=batch_size, shuffle=shuffle)
        dataloaders[directory_name] = dataloader
    return dataloaders


# =============================================================================
# Probe Model Creation
# =============================================================================

def create_probe(input_size, output_size, use_mlp=False):
    if use_mlp:
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 3),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size // 3, output_size)
        ).cuda()
    else:
        return torch.nn.Linear(input_size, output_size).cuda()


# =============================================================================
# Results Processing and Aggregation
# =============================================================================
    
def get_best_results(results):
    """
    Extract best results for each layer based on validation loss.
    
    For each layer, finds the learning rate that achieved the best validation
    performance and extracts the corresponding metrics at that epoch.
    """
    best_results = defaultdict(list)
    for level, lrs in results.items():
        for lr, values in lrs.items():
            if "test_loss" in values:
                # test_loss is stored as (loss_value, best_epoch_index)
                best_epoch = values["test_loss"][1]
                best_results["best_lr"].append(lr)
                best_results["best_epoch"].append(best_epoch)
                best_results["train_losses"].append(values['train_losses'][best_epoch])
                best_results["train_accuracies"].append(values['train_accuracies'][best_epoch])
                best_results["val_losses"].append(values['val_losses'][best_epoch])
                best_results["val_accuracies"].append(values['val_accuracies'][best_epoch])
                best_results["test_losses"].append(values['test_loss'][0])
                best_results["test_accuracies"].append(values['test_accuracy'][0])
                break

    return best_results


def aggregate_best_results(best_results_list):
    """
    Aggregate best results across multiple experimental runs.
    
    Combines results from multiple runs by:
    - Collecting unique values for categorical metrics (best_lr, best_epoch)
    - Averaging numerical metrics (losses, accuracies)
    - Computing standard deviation for accuracy metrics
    """
    aggregated = defaultdict(list)
    if not best_results_list:
        return aggregated

    # Categorical metrics: collect unique values across runs
    set_keys = ["best_lr", "best_epoch"]
    # Numerical metrics: compute mean across runs
    avg_keys = ["train_losses", "train_accuracies",
                "val_losses", "val_accuracies",
                "test_losses", "test_accuracies"]
    # Accuracy metrics: also compute standard deviation
    std_keys = ["train_accuracies", "val_accuracies", "test_accuracies"]

    # Determine the number of layers from first run
    length = len(best_results_list[0]["best_lr"])

    # Aggregate per layer index
    for i in range(length):
        for k in set_keys:
            vals = set()
            for d in best_results_list:
                if k in d and len(d[k]) > i:
                    vals.add(d[k][i])
            aggregated[k].append(list(vals))

        for k in avg_keys:
            vals = []
            for d in best_results_list:
                if k in d and len(d[k]) > i:
                    vals.append(d[k][i])
            if vals:
                aggregated[k].append(sum(vals) / len(vals))
                # Compute standard deviation for accuracy metrics
                if k in std_keys:
                    mean = sum(vals) / len(vals)
                    variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                    std = variance ** 0.5
                    aggregated[k.replace("accuracies", "accuracies_std")].append(std)
            else:
                aggregated[k].append(float('nan'))
                if k in std_keys:
                    aggregated[k.replace("accuracies", "accuracies_std")].append(float('nan'))

    return aggregated


# =============================================================================
# Results Saving and Visualization
# =============================================================================

def save_and_plot_results(results, args, order_layers, best_probes, category):
    """
    Save experimental results to disk and generate visualization plots.
    
    Saves:
    - results.json: Full results dictionary
    - best_probes/: Trained probe model weights for each layer
    - log.txt: Human-readable summary of results
    - Various plots showing loss/accuracy curves across model layers
    
    The plots visualize how probe performance varies across different
    model components (vision encoder, projector, language model) using
    colored background regions.
    """
    # Construct save path based on experiment configuration
    if args.use_mlp:
        save_path = args.save_path + "/mlp"
    else:
        save_path = args.save_path + "/linear"
    if args.random_baseline:
        save_path = save_path + "/random_baseline"
    save_path = os.path.join(save_path, category, args.parent_features_directory.split("features/")[-1], f"distance_{args.distance}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save raw results as JSON
    with open(save_path + "/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    # Save trained probe models for each layer and run
    for level, value in best_probes.items():
        for i, probe in enumerate(value):
            os.makedirs(os.path.join(save_path, f"best_probes/{level.replace('/', '_')}"), exist_ok=True)
            torch.save(probe, os.path.join(save_path, f"best_probes/{level.replace('/', '_')}/run{i}.pt"))

    # Generate human-readable log file
    log = ""
    for i in range(len(results["best_lr"])):
        log += f"{order_layers[i]}:\n"
        log += f"\t Best LR: {results['best_lr'][i]}\n"
        log += f"\t Best Epoch: {results['best_epoch'][i]}\n"
        log += f"\t Train Loss: {results['train_losses'][i]:.4f}, Train Acc: {results['train_accuracies'][i]:.4f} (std: {results['train_accuracies_std'][i]:.4f})\n"
        log += f"\t Validation Loss: {results['val_losses'][i]:.4f}, Validation Acc: {results['val_accuracies'][i]:.4f} (std: {results['val_accuracies_std'][i]:.4f})\n"
        log += f"\t Test Loss: {results['test_losses'][i]:.4f}, Test Acc: {results['test_accuracies'][i]:.4f} (std: {results['test_accuracies_std'][i]:.4f})\n"
        log += "----------------------------------------\n"

    with open(save_path + "/log.txt", 'w') as f:
        f.write(log)

    # Calculate layer boundaries for colored background regions in plots
    vis_enc_layer = 0
    plot_projector = False
    for layer in order_layers:
        if "vision" in layer:
            vis_enc_layer += 1
            llm_layer = vis_enc_layer
        elif layer == "projector":
            plot_projector = True
            proj_layer = vis_enc_layer + 1
            llm_layer = proj_layer
        elif "language" in layer:
            llm_layer += 1

    # --- Plot 1: Train/Validation Loss across layers ---
    plt.plot(range(1, len(results["train_losses"]) + 1), results["train_losses"], label="Train Loss")
    plt.plot(range(1, len(results["val_losses"]) + 1), results["val_losses"], label="Validation Loss")
    plt.xlabel("Layers")
    plt.ylabel("Loss")
    # Add colored background regions to indicate model components
    plt.axvspan(1, vis_enc_layer, facecolor="#90ee90", alpha=0.3, label="Vision Encoder")
    if plot_projector:
        plt.axvspan(vis_enc_layer, proj_layer, facecolor="#ff7f7f", alpha=0.3, label="Projector")
        plt.axvspan(proj_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    else:
        plt.axvspan(vis_enc_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  
        ncol=3                       
    )
    plt.savefig(save_path + "/trainval_loss_curves.png", bbox_inches="tight")
    plt.close()

    # --- Plot 2: Train/Validation Accuracy across layers ---
    m = min(results["train_accuracies"] + results["val_accuracies"])
    plt.plot(range(1, len(results["train_accuracies"]) + 1), results["train_accuracies"], label="Train Accuracy")
    plt.plot(range(1, len(results["val_accuracies"]) + 1), results["val_accuracies"], label="Validation Accuracy")
    plt.xlabel("Layers")
    plt.ylabel("Accuracy")
    plt.ylim(m, 1)
    # Add colored background regions for model components
    plt.axvspan(1, vis_enc_layer, facecolor="#90ee90", alpha=0.3, label="Vision Encoder")
    if plot_projector:
        plt.axvspan(vis_enc_layer, proj_layer, facecolor="#ff7f7f", alpha=0.3, label="Projector")
        plt.axvspan(proj_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    else:
        plt.axvspan(vis_enc_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  
        ncol=3                       
    )
    plt.savefig(save_path + "/trainval_accuracy_curves.png", bbox_inches="tight")
    plt.close()

    # --- Plot 3: Test Loss across layers ---
    plt.plot(range(1, len(results["test_losses"]) + 1), results["test_losses"])
    plt.xlabel("Layers")
    plt.ylabel("Test Loss")
    # Add colored background regions for model components
    plt.axvspan(1, vis_enc_layer, facecolor="#90ee90", alpha=0.3, label="Vision Encoder")
    if plot_projector:
        plt.axvspan(vis_enc_layer, proj_layer, facecolor="#ff7f7f", alpha=0.3, label="Projector")
        plt.axvspan(proj_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    else:
        plt.axvspan(vis_enc_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  
        ncol=3                       
    )
    plt.savefig(save_path + "/test_loss_curve.png", bbox_inches="tight")
    plt.close()

    # --- Plot 4: Test Accuracy across layers ---
    plt.plot(range(1, len(results["test_accuracies"]) + 1), results["test_accuracies"])
    plt.xlabel("Layers")
    plt.ylabel("Test Accuracy")
    plt.ylim(min(results["test_accuracies"]), 1)  # Set y-axis from minimum accuracy to 1.0
    # Add colored background regions for model components
    plt.axvspan(1, vis_enc_layer, facecolor="#90ee90", alpha=0.3, label="Vision Encoder")
    if plot_projector:
        plt.axvspan(vis_enc_layer, proj_layer, facecolor="#ff7f7f", alpha=0.3, label="Projector")
        plt.axvspan(proj_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    else:
        plt.axvspan(vis_enc_layer, llm_layer, facecolor="#87cefa", alpha=0.3, label="LLM")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),  
        ncol=3                       
    )
    plt.savefig(save_path + "/test_accuracy_curve.png", bbox_inches="tight")
    plt.close()
