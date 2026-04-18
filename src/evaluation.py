import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_main_process
from flow import MultiModalDOPRI5Solver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def denormalize_image(image_tensor):
    """Convert image from [-1,1] to [0,1] range."""
    return (image_tensor + 1) / 2

def prepare_visualization_data(batch, indices, device):
    """Extract and prepare data for visualization."""
    control_rna = batch['control_transcriptomics'][indices].to(device)
    real_treatment_images = batch['treatment_images'][indices].to(device)
    control_images = batch['control_images'][indices].to(device)

    conditioning_info = batch['conditioning_info']
    conditioning_subset = {}
    
    if isinstance(conditioning_info, list):
        # Handle list format from dataloader
        selected_conditioning = [conditioning_info[i] for i in indices]
        conditioning_subset = {
            'treatment': [info['treatment'] for info in selected_conditioning],
            'cell_line': [info['cell_line'] for info in selected_conditioning],
            'compound_concentration_in_uM': [info['compound_concentration_in_uM'] for info in selected_conditioning],
            'timepoint': [info['timepoint'] for info in selected_conditioning]
        }
    else:
        # Handle dict format
        for key, values in conditioning_info.items():
            if isinstance(values, list):
                conditioning_subset[key] = [values[i] for i in indices]
            else:
                conditioning_subset[key] = [values[i].item() for i in indices]
    
    return control_rna, real_treatment_images, control_images, conditioning_subset

def create_comparison_plot(control_images, predicted_images, real_images, conditioning_info, num_samples, output_path):
    """Create and save comparison visualization."""
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Control image
        control_rgb = denormalize_image(control_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[0, i].imshow(control_rgb)
        axes[0, i].set_title(f"Control\n{conditioning_info['cell_line'][i]}")
        axes[0, i].axis('off')
        
        # Predicted treatment image
        pred_rgb = denormalize_image(predicted_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[1, i].imshow(pred_rgb)
        axes[1, i].set_title(f"Predicted Treatment\n{conditioning_info['treatment'][i]}")
        axes[1, i].axis('off')
        
        # Real treatment image
        real_rgb = denormalize_image(real_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[2, i].imshow(real_rgb)
        axes[2, i].set_title(f"Real Treatment\n{conditioning_info['compound_concentration_in_uM'][i]:.1f}μM")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_conditioning_batch(conditioning_info: Dict, 
                              compound_to_idx: Dict, 
                              cell_line_to_idx: Dict,
                              device: torch.device,
                              drug_embeddings: Dict = None) -> Dict[str, torch.Tensor]:
    """Convert conditioning info to tensors."""
    if isinstance(conditioning_info, list):
        # Handle list format from dataloader
        batch_size = len(conditioning_info)
        treatments = [info['treatment'] for info in conditioning_info]
        cell_lines = [info['cell_line'] for info in conditioning_info]
        concentrations = [info['compound_concentration_in_uM'] for info in conditioning_info]
        timepoints = [info['timepoint'] for info in conditioning_info]
    else:
        # Handle dict format
        batch_size = len(conditioning_info['treatment'])
        treatments = conditioning_info['treatment']
        cell_lines = conditioning_info['cell_line']
        concentrations = conditioning_info['compound_concentration_in_uM']
        timepoints = conditioning_info['timepoint']
    
    # Convert categorical variables to indices
    compound_ids = torch.tensor([
        compound_to_idx[comp] for comp in treatments
    ], device=device)
    
    cell_line_ids = torch.tensor([
        cell_line_to_idx[cl] for cl in cell_lines
    ], device=device)
    
    # Convert continuous variables
    concentration_tensor = torch.as_tensor(
        concentrations, dtype=torch.float32, device=device
    )

    timepoint_tensor = torch.as_tensor(
        timepoints, dtype=torch.float32, device=device
    )
    
    result = {
        'compound_ids': compound_ids,
        'cell_line_ids': cell_line_ids,
        'concentration': concentration_tensor,
        'timepoint': timepoint_tensor
    }
    
    # Add drug embeddings if provided
    if drug_embeddings is not None:
        result['drug_embeddings'] = drug_embeddings
    
    return result

def generate_multimodal_drug_conditioned_outputs(
    model, rectified_flow, control_rna, control_images, conditioning_info,
    compound_to_idx, cell_line_to_idx, device, num_steps=100):
    """Generate both treatment transcriptome and treatment images using clean multimodal solver."""
    
    # Model should already be unwrapped when passed here
    model.eval()
    
    control_rna = control_rna.to(device)
    control_images = control_images.to(device)
    
    conditioning_tensors = prepare_conditioning_batch(
        conditioning_info, compound_to_idx, cell_line_to_idx, device, drug_embeddings=None
    )
    
    # Generate treatment transcriptome
    with torch.no_grad():
        treatment_transcriptome = model(
            x=None, t=None,
            control_rna=control_rna,
            control_images=control_images,
            conditioning_info=conditioning_tensors,
            mode='transcriptome_direct'
        )
    
    # Generate treatment images
    solver = MultiModalDOPRI5Solver(
        model=model,
        rectified_flow=rectified_flow,
        control_rna=control_rna,
        control_images=control_images,
        conditioning_info=conditioning_tensors
    )
    
    treatment_images = solver.generate_sample(num_steps=num_steps, device=device)
    treatment_images = torch.clamp(treatment_images, -1, 1)
    
    return treatment_transcriptome, treatment_images

def evaluate_drug_effects(model, rectified_flow, val_loader, compound_to_idx, 
                         cell_line_to_idx, device, output_dir, num_samples=10):
    """Evaluate and visualize drug effects (only on main process)."""
    if not is_main_process():
        return
    
    # Model should already be unwrapped when passed here
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    batch = next(iter(val_loader))
    batch_size = len(batch['control_transcriptomics'])
    actual_num_samples = min(num_samples, batch_size)
    indices = torch.randperm(batch_size)[:actual_num_samples]
    
    control_rna, real_treatment_images, control_images, conditioning_subset = prepare_visualization_data(
        batch, indices, device
    )
    
    with torch.no_grad():
        predicted_treatment_rna, predicted_treatment_images = generate_multimodal_drug_conditioned_outputs(
            model, rectified_flow, control_rna, control_images, conditioning_subset,
            compound_to_idx, cell_line_to_idx, device
        )

    output_path = os.path.join(output_dir, "drug_effect_comparison.png")
    create_comparison_plot(
        control_images, predicted_treatment_images, real_treatment_images,
        conditioning_subset, actual_num_samples, output_path
    )
    
    logger.info(f"Drug effect evaluation saved to {output_dir}")
    logger.info(f"Evaluated {actual_num_samples} samples (requested {num_samples})")

def evaluate_cross_modal_consistency(model, val_loader, compound_to_idx, cell_line_to_idx, device):
    """
    Quantitatively evaluate cross-modal consistency between RNA and image predictions.
    Model can be either DDP-wrapped or unwrapped - handles both cases.
    """
    # Only run on main process to avoid distributed issues
    if not is_main_process():
        return {}
        
    # Get the actual model (unwrap if DDP) for direct access
    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.eval()
    
    all_rna_features = []
    all_image_features = []
    all_pred_rna = []
    all_real_rna = []
    alignment_scores = []
    
    with torch.no_grad():
        # Iterate through the entire validation dataset
        for batch in val_loader:
            conditioning_tensors = prepare_conditioning_batch(
                batch['conditioning_info'], compound_to_idx, cell_line_to_idx, device, drug_embeddings=None
            )
            
            control_rna = batch['control_transcriptomics'].to(device).float()
            control_images = batch['control_images'].to(device).float()
            real_treatment_rna = batch['treatment_transcriptomics'].to(device).float()
            
            # Use the actual model (unwrapped) for evaluation
            # Get shared representations and features
            representations = actual_model(
                x=None, t=None,
                control_rna=control_rna,
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='shared'
            )
            
            # Get RNA predictions
            pred_rna = actual_model(
                x=None, t=None,
                control_rna=control_rna,
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='transcriptome_direct'
            )
            
            # Extract features for correlation analysis
            rna_enhanced = representations['rna_enhanced']
            image_enhanced = representations['image_enhanced']
            
            # Compute cross-modal alignment score for this batch
            alignment_score = compute_cross_modal_alignment_score(rna_enhanced, image_enhanced)
            alignment_scores.append(alignment_score.item())
            
            # Collect features and predictions
            all_rna_features.append(rna_enhanced.cpu())
            all_image_features.append(image_enhanced.cpu())
            all_pred_rna.append(pred_rna.cpu())
            all_real_rna.append(real_treatment_rna.cpu())
    
    # Concatenate all batches
    all_rna_features = torch.cat(all_rna_features, dim=0)
    all_image_features = torch.cat(all_image_features, dim=0)
    all_pred_rna = torch.cat(all_pred_rna, dim=0)
    all_real_rna = torch.cat(all_real_rna, dim=0)
    
    # Compute comprehensive metrics
    metrics = {
        'cross_modal_alignment': np.mean(alignment_scores),
        'rna_prediction_mse': F.mse_loss(all_pred_rna, all_real_rna).item(),
        'feature_correlation': compute_feature_correlation(all_rna_features, all_image_features),
        'shared_variance_explained': compute_shared_variance(all_rna_features, all_image_features),
        'consistency_score': compute_consistency_score(all_pred_rna, all_real_rna, all_rna_features)
    }
    
    return metrics

def compute_cross_modal_alignment_score(rna_features, image_features, temperature=0.1):
    """Compute cosine similarity alignment score (matching training)."""
    # Normalize features
    rna_norm = F.normalize(rna_features, p=2, dim=1)
    image_norm = F.normalize(image_features, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(rna_norm, image_norm, dim=1)
    
    # Return similarity (higher = better aligned)
    return cosine_sim.mean()

def compute_feature_correlation(rna_features, image_features):
    """Compute correlation between RNA and image feature representations."""
    # Flatten features
    rna_flat = rna_features.view(rna_features.shape[0], -1)
    image_flat = image_features.view(image_features.shape[0], -1)
    
    # Compute correlation matrix
    rna_centered = rna_flat - rna_flat.mean(dim=0, keepdim=True)
    image_centered = image_flat - image_flat.mean(dim=0, keepdim=True)
    
    # Correlation coefficient
    correlation = torch.sum(rna_centered * image_centered, dim=0) / (
        torch.sqrt(torch.sum(rna_centered ** 2, dim=0)) * 
        torch.sqrt(torch.sum(image_centered ** 2, dim=0)) + 1e-8
    )
    
    return torch.mean(torch.abs(correlation)).item()

def compute_shared_variance(rna_features, image_features):
    """Compute how much variance is shared between modalities."""
    # Use canonical correlation analysis approximation
    rna_flat = rna_features.view(rna_features.shape[0], -1)
    image_flat = image_features.view(image_features.shape[0], -1)
    
    # Center the data
    rna_centered = rna_flat - rna_flat.mean(dim=0, keepdim=True)
    image_centered = image_flat - image_flat.mean(dim=0, keepdim=True)
    
    # Cross-covariance matrix
    cross_cov = torch.mm(rna_centered.t(), image_centered) / (rna_features.shape[0] - 1)
    
    # Frobenius norm of cross-covariance (proxy for shared variance)
    shared_variance = torch.norm(cross_cov, 'fro').item()
    
    return shared_variance

def compute_consistency_score(pred_rna, real_rna, rna_features):
    """Compute correlation between predicted and real RNA."""
    # Flatten tensors
    pred_flat = pred_rna.view(-1)
    real_flat = real_rna.view(-1)
    
    # Compute Pearson correlation
    pred_centered = pred_flat - pred_flat.mean()
    real_centered = real_flat - real_flat.mean()
    
    correlation = torch.sum(pred_centered * real_centered) / (
        torch.sqrt(torch.sum(pred_centered ** 2)) * 
        torch.sqrt(torch.sum(real_centered ** 2)) + 1e-8
    )
    
    return correlation.item()

def evaluate_rna_only_performance(model, val_loader, compound_to_idx, cell_line_to_idx, device, output_dir):
    """Evaluate RNA-only model performance."""
    if not is_main_process():
        return
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_pred_rna = []
    all_real_rna = []
    all_mse_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            conditioning_tensors = prepare_conditioning_batch(
                batch['conditioning_info'], compound_to_idx, cell_line_to_idx, device, drug_embeddings=None
            )
            
            control_rna = batch['control_transcriptomics'].to(device).float()
            real_treatment_rna = batch['treatment_transcriptomics'].to(device).float()
            
            # RNA-only prediction
            pred_rna = model(
                control_rna=control_rna,
                conditioning_info=conditioning_tensors,
                mode='transcriptome_direct'
            )
            
            # Compute batch MSE
            batch_mse = F.mse_loss(pred_rna, real_treatment_rna).item()
            all_mse_losses.append(batch_mse)
            
            # Collect for overall correlation analysis
            all_pred_rna.append(pred_rna.cpu())
            all_real_rna.append(real_treatment_rna.cpu())
    
    # Concatenate all predictions
    all_pred_rna = torch.cat(all_pred_rna, dim=0)
    all_real_rna = torch.cat(all_real_rna, dim=0)
    
    # Compute metrics
    overall_mse = F.mse_loss(all_pred_rna, all_real_rna).item()
    
    # Compute correlation
    pred_flat = all_pred_rna.view(-1)
    real_flat = all_real_rna.view(-1)
    pred_centered = pred_flat - pred_flat.mean()
    real_centered = real_flat - real_flat.mean()
    correlation = torch.sum(pred_centered * real_centered) / (
        torch.sqrt(torch.sum(pred_centered ** 2)) * 
        torch.sqrt(torch.sum(real_centered ** 2)) + 1e-8
    )
    
    metrics = {
        'rna_mse': overall_mse,
        'rna_correlation': correlation.item(),
        'num_samples': len(all_pred_rna),
    }
    
    logger.info("=== RNA-Only Model Performance ===")
    logger.info(f"RNA MSE: {metrics['rna_mse']:.6f}")
    logger.info(f"RNA Correlation: {metrics['rna_correlation']:.4f}")
    logger.info(f"Evaluated on {metrics['num_samples']} samples")
    
    # Save metrics
    import json
    metrics_path = os.path.join(output_dir, "rna_only_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"RNA-only metrics saved to {metrics_path}")
    return metrics

def evaluate_image_only_performance(model, rectified_flow, val_loader, compound_to_idx, 
                                  cell_line_to_idx, device, output_dir, num_steps=100):
    """Evaluate Image-only model performance."""
    if not is_main_process():
        return
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Take a few samples for visual evaluation
    batch = next(iter(val_loader))
    batch_size = len(batch['control_transcriptomics'])
    num_samples = min(8, batch_size)
    indices = torch.randperm(batch_size)[:num_samples]
    
    control_images = batch['control_images'][indices].to(device)
    real_treatment_images = batch['treatment_images'][indices].to(device)
    
    conditioning_subset = {}
    for key, values in batch['conditioning_info'].items():
        if isinstance(values, list):
            conditioning_subset[key] = [values[i] for i in indices]
        else:
            conditioning_subset[key] = [values[i].item() for i in indices]
    
    conditioning_tensors = prepare_conditioning_batch(
        conditioning_subset, compound_to_idx, cell_line_to_idx, device, drug_embeddings=None
    )
    
    with torch.no_grad():
        # Generate treatment images using DOPRI5 solver
        solver = MultiModalDOPRI5Solver(
            model=model,
            rectified_flow=rectified_flow,
            control_rna=None,  # Not used in image-only model
            control_images=control_images,
            conditioning_info=conditioning_tensors
        )
        
        predicted_treatment_images = solver.generate_sample(num_steps=num_steps, device=device)
        predicted_treatment_images = torch.clamp(predicted_treatment_images, -1, 1)
    
    # Create comparison visualization
    output_path = os.path.join(output_dir, "image_only_comparison.png")
    create_image_only_comparison_plot(
        control_images, predicted_treatment_images, real_treatment_images,
        conditioning_subset, num_samples, output_path
    )
    
    logger.info("=== Image-Only Model Performance ===")
    logger.info(f"Generated {num_samples} image samples")
    logger.info(f"Image-only evaluation saved to {output_dir}")


def create_image_only_comparison_plot(control_images, predicted_images, real_images, 
                                    conditioning_info, num_samples, output_path):
    """Create comparison plot for image-only model."""
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Control image
        control_rgb = denormalize_image(control_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[0, i].imshow(control_rgb)
        axes[0, i].set_title(f"Control\n{conditioning_info['cell_line'][i]}")
        axes[0, i].axis('off')
        
        # Predicted treatment image
        pred_rgb = denormalize_image(predicted_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[1, i].imshow(pred_rgb)
        axes[1, i].set_title(f"Predicted (Image-Only)\n{conditioning_info['treatment'][i]}")
        axes[1, i].axis('off')
        
        # Real treatment image
        real_rgb = denormalize_image(real_images[i][:3].permute(1, 2, 0).cpu().numpy())
        axes[2, i].imshow(real_rgb)
        axes[2, i].set_title(f"Real Treatment\n{conditioning_info['compound_concentration_in_uM'][i]:.1f}μM")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_ablation_results(full_model_dir, rna_only_dir, image_only_dir, output_dir):
    """Compare results across all three model types."""
    if not is_main_process():
        return
    
    import json
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Load RNA metrics
    try:
        with open(os.path.join(rna_only_dir, "rna_only_metrics.json"), 'r') as f:
            results['rna_only'] = json.load(f)
    except FileNotFoundError:
        results['rna_only'] = {}
    
    # Load cross-modal metrics (from full model)
    try:
        with open(os.path.join(full_model_dir, "cross_modal_consistency_metrics.json"), 'r') as f:
            results['full_model'] = json.load(f)
    except FileNotFoundError:
        results['full_model'] = {}
    
    # Create comparison summary
    comparison_summary = {
        "RNA Prediction Performance": {
            "RNA-Only MSE": results.get('rna_only', {}).get('rna_mse', 'N/A'),
            "RNA-Only Correlation": results.get('rna_only', {}).get('rna_correlation', 'N/A'),
            "Cross-Modal MSE": results.get('full_model', {}).get('rna_prediction_mse', 'N/A'),
            "Cross-Modal Correlation": results.get('full_model', {}).get('consistency_score', 'N/A'),
        },
        "Cross-Modal Capabilities": {
            "Cross-Modal Alignment": results.get('full_model', {}).get('cross_modal_alignment', 'N/A'),
            "Feature Correlation": results.get('full_model', {}).get('feature_correlation', 'N/A'),
            "Shared Variance": results.get('full_model', {}).get('shared_variance_explained', 'N/A'),
        }
    }
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "ablation_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info("=== Ablation Study Comparison ===")
    logger.info(f"Results saved to {comparison_path}")
    
    # Print summary
    for category, metrics in comparison_summary.items():
        logger.info(f"\n{category}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value}")
    
    return comparison_summary