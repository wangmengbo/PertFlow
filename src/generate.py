import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import RectifiedFlow, MultiModalDOPRI5Solver
from model import MultiModalDrugConditionedModel, RNAOnlyDrugConditionedModel, ImageOnlyDrugConditionedModel
from dataloader import create_dataloader, image_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_conditioning_batch(conditioning_info, 
                              compound_to_idx: dict, 
                              cell_line_to_idx: dict,
                              device: torch.device) -> dict:
    """Convert conditioning info to tensors."""
    if isinstance(conditioning_info, list):
        batch_size = len(conditioning_info)
        treatments = [info['treatment'] for info in conditioning_info]
        cell_lines = [info['cell_line'] for info in conditioning_info]
        concentrations = [info['compound_concentration_in_uM'] for info in conditioning_info]
        timepoints = [info['timepoint'] for info in conditioning_info]
    else:
        batch_size = len(conditioning_info['treatment'])
        treatments = conditioning_info['treatment']
        cell_lines = conditioning_info['cell_line']
        concentrations = conditioning_info['compound_concentration_in_uM']
        timepoints = conditioning_info['timepoint']
    
    unk_compound_idx = 0  # fallback; KG encoder encodes by structure, not this index
    unk_cell_line_idx = 0

    unseen_compounds = set(t for t in treatments if t not in compound_to_idx)
    if unseen_compounds:
        logger.info(f"Unseen compounds mapped to unk index (KG encoder will handle): {unseen_compounds}")

    compound_ids = torch.tensor([
        compound_to_idx.get(comp, unk_compound_idx) for comp in treatments
    ], device=device)
    
    cell_line_ids = torch.tensor([
        cell_line_to_idx.get(cl, unk_cell_line_idx) for cl in cell_lines
    ], device=device)
    
    concentrations = torch.as_tensor(
        concentrations, 
        dtype=torch.float32, device=device
    )

    timepoints = torch.as_tensor(
        timepoints,
        dtype=torch.float32, device=device
    )
    
    return {
        'compound_ids': compound_ids,
        'cell_line_ids': cell_line_ids,
        'concentration': concentrations,
        'timepoint': timepoints,
        'compound_names': treatments,  # pass raw names so KG encoder can look them up by structure
    }

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, ablation_type: str = 'none'):
    """Load model from checkpoint with proper configuration based on ablation type."""
    logger.info(f"Loading {ablation_type} model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    config = checkpoint['config']
    vocab_mappings = checkpoint['vocab_mappings']
    model_state_dict = checkpoint['model']
    
    # Infer missing parameters from saved weights if not in config
    drug_embed_dim = config.get('drug_embed_dim')
    if drug_embed_dim is None:
        # Infer from compound embedding weights: [vocab_size, embed_dim]
        if 'drug_embedding.compound_embedding.weight' in model_state_dict:
            drug_embed_dim = model_state_dict['drug_embedding.compound_embedding.weight'].shape[1]
            logger.info(f"Inferred drug_embed_dim={drug_embed_dim} from checkpoint weights")
        else:
            drug_embed_dim = 256  # fallback to common training default
            logger.warning(f"Could not infer drug_embed_dim, using fallback: {drug_embed_dim}")
    
    shared_embed_dim = config.get('shared_embed_dim')
    if shared_embed_dim is None:
        # Infer from shared encoder weights if available
        if 'shared_encoder.0.weight' in model_state_dict:
            shared_embed_dim = model_state_dict['shared_encoder.0.weight'].shape[0]
            logger.info(f"Inferred shared_embed_dim={shared_embed_dim} from checkpoint weights")
        else:
            shared_embed_dim = 512  # fallback
            logger.warning(f"Could not infer shared_embed_dim, using fallback: {shared_embed_dim}")
    
    # Common parameters for all models
    common_params = {
        'compound_vocab_size': config['compound_vocab_size'],
        'cell_line_vocab_size': config['cell_line_vocab_size'],
        'rna_dim': config['rna_dim'],
        'img_channels': config.get('img_channels', 4),
        'img_size': config.get('img_size', 256),
        'drug_embed_dim': drug_embed_dim,
        'shared_embed_dim': shared_embed_dim,
    }
    
    # Initialize appropriate model based on ablation type
    if ablation_type == 'none':
        # Full cross-modal model
        model = MultiModalDrugConditionedModel(
            **common_params,
            model_channels=config.get('model_channels', 128),
            rna_output_dim=config.get('rna_output_dim', 256),
            gene_embed_dim=config.get('gene_embed_dim', 64),
            image_output_dim=config.get('image_output_dim', 256),
            use_kg_drug_encoder=config.get('use_kg_drug_encoder', False),
            use_kg_gene_encoder=config.get('use_kg_gene_encoder', False),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [16]),
            dropout=config.get('dropout', 0.1),
            channel_mult=config.get('channel_mult', (1, 2, 2, 2)),
            use_checkpoint=config.get('use_checkpoint', False),
            num_heads=config.get('num_heads', 4),
            num_head_channels=config.get('num_head_channels', 32),
            use_scale_shift_norm=config.get('use_scale_shift_norm', True),
            resblock_updown=config.get('resblock_updown', True),
            use_new_attention_order=config.get('use_new_attention_order', True),
        )
        logger.info("Loaded full cross-modal model")
        
    elif ablation_type == 'rna_only':
        # RNA-only model
        model = RNAOnlyDrugConditionedModel(
            **common_params,
            rna_output_dim=config.get('rna_output_dim', 256),
            gene_embed_dim=config.get('gene_embed_dim', 64),
        )
        logger.info("Loaded RNA-only model")
        
    elif ablation_type == 'image_only':
        # Image-only model
        model = ImageOnlyDrugConditionedModel(
            **common_params,
            model_channels=config.get('model_channels', 128),
            image_output_dim=config.get('image_output_dim', 256),
            num_res_blocks=config.get('num_res_blocks', 2),
            attention_resolutions=config.get('attention_resolutions', [16]),
            dropout=config.get('dropout', 0.1),
            channel_mult=config.get('channel_mult', (1, 2, 2, 2)),
            use_checkpoint=config.get('use_checkpoint', False),
            num_heads=config.get('num_heads', 4),
            num_head_channels=config.get('num_head_channels', 32),
            use_scale_shift_norm=config.get('use_scale_shift_norm', True),
            resblock_updown=config.get('resblock_updown', True),
            use_new_attention_order=config.get('use_new_attention_order', True),
        )
        logger.info("Loaded image-only model")
        
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    # Filter out dynamically created modules from checkpoint
    filtered_state_dict = {}
    
    # Keys to exclude (dynamically created during training)
    exclude_keys = [
        'rna_to_image_recon',
        'image_to_rna_recon', 
        'mi_estimator'
    ]
    
    for key, value in model_state_dict.items():
        # Check if key starts with any excluded prefix
        if not any(key.startswith(exclude_key) for exclude_key in exclude_keys):
            filtered_state_dict[key] = value
        else:
            logger.info(f"Skipping dynamically created module: {key}")
    
    # Load filtered model weights
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'Unknown')}")
    logger.info(f"Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
    
    return model, vocab_mappings['compound_to_idx'], vocab_mappings['cell_line_to_idx']

def generate_rna_predictions(model, val_loader, compound_to_idx, cell_line_to_idx, 
                           device, num_samples=50, ablation_type='none'):
    """Generate RNA predictions for all model types that support it."""
    logger.info(f"Generating RNA predictions for {num_samples} samples ({ablation_type} model)...")
    
    generated_data = {
        'control_rna': [],
        'treatment_rna_real': [],
        'treatment_rna_generated': [],
        'conditioning_info': []
    }
    
    # Add control/treatment images for full model
    if ablation_type == 'none':
        generated_data['control_images'] = []
        generated_data['treatment_images_real'] = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating RNA predictions"):
            if samples_collected >= num_samples:
                break
                
            batch_size = len(batch['control_transcriptomics'])
            samples_to_take = min(batch_size, num_samples - samples_collected)
            
            # Extract batch data
            control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
            treatment_rna_real = batch['treatment_transcriptomics'][:samples_to_take]
                        
            # Prepare conditioning
            conditioning_info = batch['conditioning_info']
            conditioning_subset = {}

            if isinstance(conditioning_info, list):
                # Handle list format from dataloader
                selected_conditioning = [conditioning_info[i] for i in range(samples_to_take)]
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
                        conditioning_subset[key] = values[:samples_to_take]
                    else:
                        conditioning_subset[key] = [values[i].item() for i in range(samples_to_take)]
            
            conditioning_tensors = prepare_conditioning_batch(
                conditioning_subset, compound_to_idx, cell_line_to_idx, device
            )
            
            # Generate treatment RNA based on model type
            if ablation_type == 'rna_only':
                treatment_rna_generated = model(
                    control_rna=control_rna,
                    conditioning_info=conditioning_tensors,
                    mode='transcriptome_direct'
                )
            elif ablation_type == 'none':
                # Full model - need images too
                control_images = batch['control_images'][:samples_to_take].to(device)
                treatment_images_real = batch['treatment_images'][:samples_to_take]
                
                treatment_rna_generated = model(
                    x=None, t=None,
                    control_rna=control_rna,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors,
                    mode='transcriptome_direct'
                )
                
                # Store image data too
                generated_data['control_images'].append(control_images.cpu())
                generated_data['treatment_images_real'].append(treatment_images_real)
            else:
                # Image-only model doesn't generate RNA
                logger.error("Image-only model cannot generate RNA predictions!")
                return None
            
            # Store results
            generated_data['control_rna'].append(control_rna.cpu())
            generated_data['treatment_rna_real'].append(treatment_rna_real)
            generated_data['treatment_rna_generated'].append(treatment_rna_generated.cpu())
            generated_data['conditioning_info'].append(conditioning_subset)
            
            samples_collected += samples_to_take
    
    # Concatenate all data
    for key in ['control_rna', 'treatment_rna_real', 'treatment_rna_generated']:
        generated_data[key] = torch.cat(generated_data[key], dim=0)
    
    # Concatenate image data for full model
    if ablation_type == 'none':
        for key in ['control_images', 'treatment_images_real']:
            generated_data[key] = torch.cat(generated_data[key], dim=0)
    
    # Flatten conditioning info
    flattened_conditioning = {}
    for key in generated_data['conditioning_info'][0].keys():
        flattened_conditioning[key] = []
        for batch_info in generated_data['conditioning_info']:
            flattened_conditioning[key].extend(batch_info[key])
    generated_data['conditioning_info'] = flattened_conditioning
    
    logger.info(f"Generated RNA predictions for {len(generated_data['control_rna'])} samples")
    return generated_data

def generate_treatment_images(model, rectified_flow, val_loader, compound_to_idx, 
                            cell_line_to_idx, device, num_samples=50, num_steps=100,
                            ablation_type='none'):
    """Generate treatment images for models that support it."""
    if ablation_type == 'rna_only':
        logger.warning("RNA-only model cannot generate images!")
        return None
        
    logger.info(f"Generating treatment images for {num_samples} samples ({ablation_type} model)...")
    
    generated_data = {
        'control_images': [],
        'treatment_images_real': [],
        'treatment_images_generated': [],
        'conditioning_info': []
    }
    
    # Add RNA data for full model
    if ablation_type == 'none':
        generated_data['control_rna'] = []
        generated_data['treatment_rna_real'] = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating images"):
            if samples_collected >= num_samples:
                break
                
            batch_size = len(batch['control_transcriptomics'])
            samples_to_take = min(batch_size, num_samples - samples_collected)
            
            # Extract batch data
            control_images = batch['control_images'][:samples_to_take].to(device)
            treatment_images_real = batch['treatment_images'][:samples_to_take]
            
            conditioning_info = batch['conditioning_info']
            conditioning_subset = {}

            if isinstance(conditioning_info, list):
                selected_conditioning = [conditioning_info[i] for i in range(samples_to_take)]
                conditioning_subset = {
                    'treatment': [info['treatment'] for info in selected_conditioning],
                    'cell_line': [info['cell_line'] for info in selected_conditioning],
                    'compound_concentration_in_uM': [info['compound_concentration_in_uM'] for info in selected_conditioning],
                    'timepoint': [info['timepoint'] for info in selected_conditioning]
                }
            else:
                for key, values in conditioning_info.items():
                    if isinstance(values, list):
                        conditioning_subset[key] = values[:samples_to_take]
                    else:
                        conditioning_subset[key] = [values[i].item() for i in range(samples_to_take)]

            conditioning_tensors = prepare_conditioning_batch(
                conditioning_subset, compound_to_idx, cell_line_to_idx, device
            )
            
            # Generate treatment images based on model type
            if ablation_type == 'image_only':
                # Image-only model
                solver = MultiModalDOPRI5Solver(
                    model=model,
                    rectified_flow=rectified_flow,
                    control_rna=None,  # Not used
                    control_images=control_images,
                    conditioning_info=conditioning_tensors
                )
            elif ablation_type == 'none':
                # Full model - need RNA too
                control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
                treatment_rna_real = batch['treatment_transcriptomics'][:samples_to_take]
                
                solver = MultiModalDOPRI5Solver(
                    model=model,
                    rectified_flow=rectified_flow,
                    control_rna=control_rna,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors
                )
                
                # Store RNA data too
                generated_data['control_rna'].append(control_rna.cpu())
                generated_data['treatment_rna_real'].append(treatment_rna_real)
            
            treatment_images_generated = solver.generate_sample(num_steps=num_steps, device=device)
            treatment_images_generated = torch.clamp(treatment_images_generated, -1, 1)
            
            # Store results
            generated_data['control_images'].append(control_images.cpu())
            generated_data['treatment_images_real'].append(treatment_images_real)
            generated_data['treatment_images_generated'].append(treatment_images_generated.cpu())
            generated_data['conditioning_info'].append(conditioning_subset)
            
            samples_collected += samples_to_take
    
    # Concatenate all data
    for key in ['control_images', 'treatment_images_real', 'treatment_images_generated']:
        generated_data[key] = torch.cat(generated_data[key], dim=0)
    
    # Concatenate RNA data for full model
    if ablation_type == 'none':
        for key in ['control_rna', 'treatment_rna_real']:
            generated_data[key] = torch.cat(generated_data[key], dim=0)
    
    # Flatten conditioning info
    flattened_conditioning = {}
    for key in generated_data['conditioning_info'][0].keys():
        flattened_conditioning[key] = []
        for batch_info in generated_data['conditioning_info']:
            flattened_conditioning[key].extend(batch_info[key])
    generated_data['conditioning_info'] = flattened_conditioning
    
    logger.info(f"Generated treatment images for {len(generated_data['control_images'])} samples")
    return generated_data

def extract_model_features(model, val_loader, compound_to_idx, cell_line_to_idx, 
                         device, num_samples=200, ablation_type='none'):
    """Extract features based on model capabilities."""
    if ablation_type != 'none':
        logger.info(f"Skipping feature extraction for {ablation_type} model (no cross-modal features)")
        return None
        
    logger.info(f"Extracting model features for {num_samples} samples (full model)...")
    
    feature_data = {
        'rna_features': [],
        'image_features': [],
        'rna_enhanced': [],
        'image_enhanced': [],
        'shared_representations': [],
        'conditioning_info': [],
        'labels': []
    }
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting features"):
            if samples_collected >= num_samples:
                break
                
            batch_size = len(batch['control_transcriptomics'])
            samples_to_take = min(batch_size, num_samples - samples_collected)
            
            control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
            control_images = batch['control_images'][:samples_to_take].to(device)
            
            # Prepare conditioning
            conditioning_info = batch['conditioning_info']
            conditioning_subset = {}

            if isinstance(conditioning_info, list):
                # Handle list format from dataloader
                selected_conditioning = [conditioning_info[i] for i in range(samples_to_take)]
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
                        conditioning_subset[key] = values[:samples_to_take]
                    else:
                        conditioning_subset[key] = [values[i].item() for i in range(samples_to_take)]
            
            conditioning_tensors = prepare_conditioning_batch(
                conditioning_subset, compound_to_idx, cell_line_to_idx, device
            )
            
            # Get shared representations and all intermediate features
            representations = model(
                x=None, t=None,
                control_rna=control_rna,
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='shared'
            )
            
            # Store features
            feature_data['rna_features'].append(representations['rna_features'].cpu())
            feature_data['image_features'].append(representations['image_features'].cpu())
            feature_data['rna_enhanced'].append(representations['rna_enhanced'].cpu())
            feature_data['image_enhanced'].append(representations['image_enhanced'].cpu())
            feature_data['shared_representations'].append(representations['shared_representation'].cpu())
            feature_data['conditioning_info'].append(conditioning_subset)
            
            # Create labels for visualization
            labels = [f"{comp}_{cl}" for comp, cl in zip(
                conditioning_subset['treatment'], conditioning_subset['cell_line']
            )]
            feature_data['labels'].extend(labels)
            
            samples_collected += samples_to_take
    
    # Concatenate tensors
    for key in ['rna_features', 'image_features', 'rna_enhanced', 'image_enhanced', 'shared_representations']:
        feature_data[key] = torch.cat(feature_data[key], dim=0).numpy()
    
    logger.info(f"Extracted features for {len(feature_data['labels'])} samples")
    return feature_data

# Keep all existing evaluation functions unchanged
def calculate_and_plot_rna_metrics(generated_data, output_dir):
    """Calculate and plot RNA comparison metrics focused on model accuracy."""
    logger.info("Calculating RNA comparison metrics...")
    
    control_rna = generated_data['control_rna'].numpy()
    treatment_rna_real = generated_data['treatment_rna_real'].numpy()
    treatment_rna_generated = generated_data['treatment_rna_generated'].numpy()
    
    n_samples = control_rna.shape[0]
    
    # Initialize metrics storage
    metrics = {
        'generated_vs_real': {  # PRIMARY: Model accuracy
            'mse': [],
            'rmse': [],
            'mae': [],
            'pearson_correlation': [],
            'spearman_correlation': []
        },
        'control_vs_real': {  # CONTEXT: Drug effect magnitude
            'mse': [],
            'rmse': [],
            'mae': [],
            'pearson_correlation': [],
            'spearman_correlation': []
        },
        'drug_effect_accuracy': {
            'change_direction_accuracy': [],
            'change_magnitude_correlation': [],
            'relative_mse': []
        }
    }
    
    # Calculate metrics for each sample
    for i in range(n_samples):
        control_sample = control_rna[i]
        real_sample = treatment_rna_real[i]
        generated_sample = treatment_rna_generated[i]
        
        # PRIMARY METRIC: Generated vs Real Treatment (Model Accuracy)
        mse_accuracy = mean_squared_error(real_sample, generated_sample)
        rmse_accuracy = np.sqrt(mse_accuracy)
        mae_accuracy = mean_absolute_error(real_sample, generated_sample)
        pearson_accuracy, _ = pearsonr(real_sample, generated_sample)
        spearman_accuracy, _ = spearmanr(real_sample, generated_sample)
        
        metrics['generated_vs_real']['mse'].append(mse_accuracy)
        metrics['generated_vs_real']['rmse'].append(rmse_accuracy)
        metrics['generated_vs_real']['mae'].append(mae_accuracy)
        metrics['generated_vs_real']['pearson_correlation'].append(pearson_accuracy)
        metrics['generated_vs_real']['spearman_correlation'].append(spearman_accuracy)
        
        # CONTEXT METRIC: Control vs Real Treatment (Drug Effect Size)
        mse_effect = mean_squared_error(control_sample, real_sample)
        rmse_effect = np.sqrt(mse_effect)
        mae_effect = mean_absolute_error(control_sample, real_sample)
        pearson_effect, _ = pearsonr(control_sample, real_sample)
        spearman_effect, _ = spearmanr(control_sample, real_sample)
        
        metrics['control_vs_real']['mse'].append(mse_effect)
        metrics['control_vs_real']['rmse'].append(rmse_effect)
        metrics['control_vs_real']['mae'].append(mae_effect)
        metrics['control_vs_real']['pearson_correlation'].append(pearson_effect)
        metrics['control_vs_real']['spearman_correlation'].append(spearman_effect)
        
        # DRUG EFFECT ACCURACY: How well model captures changes
        real_change = real_sample - control_sample
        generated_change = generated_sample - control_sample
        
        # Direction accuracy: What fraction of genes changed in the right direction?
        real_direction = np.sign(real_change)
        generated_direction = np.sign(generated_change)
        direction_accuracy = np.mean(real_direction == generated_direction)
        metrics['drug_effect_accuracy']['change_direction_accuracy'].append(direction_accuracy)
        
        # Change magnitude correlation
        change_corr, _ = pearsonr(real_change, generated_change)
        metrics['drug_effect_accuracy']['change_magnitude_correlation'].append(change_corr)
        
        # Relative MSE: MSE of changes normalized by effect size
        change_mse = mean_squared_error(real_change, generated_change)
        effect_magnitude = np.var(real_change)
        relative_mse = change_mse / (effect_magnitude + 1e-8)  # Avoid division by zero
        metrics['drug_effect_accuracy']['relative_mse'].append(relative_mse)
    
    # Calculate summary statistics
    summary_metrics = {}
    for comparison in ['generated_vs_real', 'control_vs_real']:
        summary_metrics[comparison] = {}
        for metric_name in ['mse', 'rmse', 'mae', 'pearson_correlation', 'spearman_correlation']:
            values = metrics[comparison][metric_name]
            summary_metrics[comparison][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Add drug effect accuracy summary
    summary_metrics['drug_effect_accuracy'] = {}
    for metric_name in ['change_direction_accuracy', 'change_magnitude_correlation', 'relative_mse']:
        values = metrics['drug_effect_accuracy'][metric_name]
        summary_metrics['drug_effect_accuracy'][metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, 'rna_comparison_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    
    # Create main comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    metric_names = ['MSE', 'RMSE', 'MAE', 'Pearson Correlation', 'Spearman Correlation']
    metric_keys = ['mse', 'rmse', 'mae', 'pearson_correlation', 'spearman_correlation']
    
    for i, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
        row = i // 3
        col = i % 3
        
        if i < 5:  # Only 5 metrics
            # PRIMARY: Generated vs Real (Model accuracy)
            accuracy_values = metrics['generated_vs_real'][metric_key]
            # CONTEXT: Control vs Real (Drug effect size)
            effect_values = metrics['control_vs_real'][metric_key]
            
            # Box plot comparison
            data_to_plot = [accuracy_values, effect_values]
            bp = axes[row, col].boxplot(data_to_plot, 
                                       labels=['Generated vs Real\n(Model Accuracy)', 
                                              'Control vs Real\n(Drug Effect Size)'], 
                                       patch_artist=True)
            
            # Color the boxes
            bp['boxes'][0].set_facecolor('lightgreen')  # Green for accuracy (main metric)
            bp['boxes'][1].set_facecolor('lightblue')   # Blue for context
            
            axes[row, col].set_title(f'{metric_name}')
            axes[row, col].set_ylabel(metric_name)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add mean values as text
            mean_accuracy = np.mean(accuracy_values)
            mean_effect = np.mean(effect_values)
            axes[row, col].text(0.32, 0.98, f'Model Accuracy: {mean_accuracy:.4f}', 
                               transform=axes[row, col].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            axes[row, col].text(0.32, 0.85, f'Drug Effect: {mean_effect:.4f}', 
                               transform=axes[row, col].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            # Hide the extra subplot
            axes[row, col].axis('off')
    
    # Hide the last subplot (1,2)
    axes[1, 2].axis('off')
    
    plt.suptitle('RNA Prediction Accuracy vs Drug Effect Magnitude', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rna_comparison_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create drug effect accuracy plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), dpi=300)

    # Direction accuracy
    direction_acc = metrics['drug_effect_accuracy']['change_direction_accuracy']
    axes[0].hist(direction_acc, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0].axvline(np.mean(direction_acc), color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Direction Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Gene Change Direction Accuracy\nMean: {np.mean(direction_acc):.3f}')
    axes[0].grid(True, alpha=0.3)
    
    # Change magnitude correlation
    change_corr = metrics['drug_effect_accuracy']['change_magnitude_correlation']
    axes[1].hist(change_corr, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1].axvline(np.mean(change_corr), color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Change Magnitude Correlation')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Drug Effect Magnitude Correlation\nMean: {np.mean(change_corr):.3f}')
    axes[1].grid(True, alpha=0.3)
    
    # Relative MSE
    rel_mse = metrics['drug_effect_accuracy']['relative_mse']
    axes[2].hist(rel_mse, bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[2].axvline(np.mean(rel_mse), color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Relative MSE')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Relative Change Prediction Error\nMean: {np.mean(rel_mse):.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Drug Effect Prediction Accuracy Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drug_effect_accuracy_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Enhanced logging with proper interpretation
    logger.info("=" * 60)
    logger.info("RNA PREDICTION EVALUATION RESULTS")
    logger.info("=" * 60)
    
    # Model accuracy metrics (most important)
    accuracy_pearson = summary_metrics['generated_vs_real']['pearson_correlation']['mean']
    accuracy_spearman = summary_metrics['generated_vs_real']['spearman_correlation']['mean']
    accuracy_mse = summary_metrics['generated_vs_real']['mse']['mean']
    
    logger.info("MODEL ACCURACY (Generated vs Real Treatment):")
    logger.info(f"  Pearson Correlation: {accuracy_pearson:.4f} ± {summary_metrics['generated_vs_real']['pearson_correlation']['std']:.4f}")
    logger.info(f"  Spearman Correlation: {accuracy_spearman:.4f} ± {summary_metrics['generated_vs_real']['spearman_correlation']['std']:.4f}")
    logger.info(f"  MSE: {accuracy_mse:.4f} ± {summary_metrics['generated_vs_real']['mse']['std']:.4f}")
    
    # Drug effect accuracy
    direction_acc_mean = summary_metrics['drug_effect_accuracy']['change_direction_accuracy']['mean']
    change_corr_mean = summary_metrics['drug_effect_accuracy']['change_magnitude_correlation']['mean']
    
    logger.info("\nDRUG EFFECT PREDICTION ACCURACY:")
    logger.info(f"  Direction Accuracy: {direction_acc_mean:.4f} ± {summary_metrics['drug_effect_accuracy']['change_direction_accuracy']['std']:.4f}")
    logger.info(f"  Change Magnitude Correlation: {change_corr_mean:.4f} ± {summary_metrics['drug_effect_accuracy']['change_magnitude_correlation']['std']:.4f}")

def create_rna_umap_plot(generated_data, output_dir):
    """Create UMAP plots comparing control vs real treatment and generated vs real treatment."""
    logger.info("Creating RNA UMAP plots...")
   
    # Extract RNA data
    control_rna = generated_data['control_rna'].numpy()
    treatment_rna_real = generated_data['treatment_rna_real'].numpy()
    treatment_rna_generated = generated_data['treatment_rna_generated'].numpy()
   # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # First subplot: Control vs Real Treatment
    control_vs_real = np.vstack([control_rna, treatment_rna_real])
    labels_1 = ['Control'] * len(control_rna) + ['Treatment (Real)'] * len(treatment_rna_real)
   
    # Standardize and apply UMAP for first comparison
    scaler_1 = StandardScaler()
    control_vs_real_scaled = scaler_1.fit_transform(control_vs_real)
   
    logger.info("Computing UMAP embedding for Control vs Real Treatment...")
    reducer_1 = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    embedding_1 = reducer_1.fit_transform(control_vs_real_scaled)
   
    # Plot first comparison
    control_mask = np.array(labels_1) == 'Control'
    real_mask = np.array(labels_1) == 'Treatment (Real)'
   
    ax1.scatter(embedding_1[control_mask, 0], embedding_1[control_mask, 1],
               c='blue', label='Control', alpha=0.8, s=8)
    ax1.scatter(embedding_1[real_mask, 0], embedding_1[real_mask, 1],
               c='red', label='Treatment (Real)', alpha=0.8, s=8)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Control vs Real Treatment RNA Expression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
   
    # Second subplot: Generated vs Real Treatment
    generated_vs_real = np.vstack([treatment_rna_generated, treatment_rna_real])
    labels_2 = ['Treatment (Generated)'] * len(treatment_rna_generated) + ['Treatment (Real)'] * len(treatment_rna_real)
   
    # Standardize and apply UMAP for second comparison
    scaler_2 = StandardScaler()
    generated_vs_real_scaled = scaler_2.fit_transform(generated_vs_real)
   
    logger.info("Computing UMAP embedding for Generated vs Real Treatment...")
    reducer_2 = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    embedding_2 = reducer_2.fit_transform(generated_vs_real_scaled)
   
    # Plot second comparison
    generated_mask = np.array(labels_2) == 'Treatment (Generated)'
    real_mask_2 = np.array(labels_2) == 'Treatment (Real)'
   
    ax2.scatter(embedding_2[generated_mask, 0], embedding_2[generated_mask, 1],
               c='orange', label='Treatment (Generated)', alpha=0.8, s=20)
    ax2.scatter(embedding_2[real_mask_2, 0], embedding_2[real_mask_2, 1],
               c='red', label='Treatment (Real)', alpha=0.8, s=20)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title('Generated vs Real Treatment RNA Expression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
   
    # Save plot
    plt.savefig(os.path.join(output_dir, 'rna_umap_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
   
    logger.info("RNA UMAP plots saved")

def create_model_features_umap_plot(feature_data, output_dir):
    """Create UMAP plots for different model features."""
    logger.info("Creating model features UMAP plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), dpi=300)

    features_to_plot = [
        ('rna_features', 'RNA Features (Before Cross-Modal)'),
        ('rna_enhanced', 'RNA Enhanced (After Cross-Modal)'),
        ('shared_representations', 'Shared Representations'),
        ('image_features', 'Image Features (Before Cross-Modal)'),
        ('image_enhanced', 'Image Enhanced (After Cross-Modal)'),
    ]

    # Get unique compounds for coloring
    compounds = []
    cell_lines = []
    for batch_info in feature_data['conditioning_info']:
        compounds.extend(batch_info['treatment'])
        cell_lines.extend(batch_info['cell_line'])

    unique_compounds = list(set(compounds))
    compound_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_compounds)))
    compound_to_color = dict(zip(unique_compounds, compound_colors))
    sample_colors = [compound_to_color[comp] for comp in compounds]

    # Create colors for each sample
    colors = [compound_to_color[comp.split('_')[0]] for comp in feature_data['labels']]

    # Mapping subplot positions
    subplot_positions = {
        'rna_features': (0, 0),
        'rna_enhanced': (1, 0),
        'shared_representations': (2, 0),
        'image_features': (0, 1),
        'image_enhanced': (1, 1),
    }

    for feature_name, title in features_to_plot:
        row, col = subplot_positions[feature_name]

        # Standardize features
        features = feature_data[feature_name]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply UMAP
        logger.info(f"Computing UMAP for {feature_name}...")
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(features_scaled)

        # Plot
        axes[row, col].scatter(embedding[:, 0], embedding[:, 1],
                            c=colors, alpha=0.8, s=20)
        axes[row, col].set_xlabel('UMAP 1')
        axes[row, col].set_ylabel('UMAP 2')
        axes[row, col].set_title(title)
        axes[row, col].grid(True, alpha=0.3)

    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8, label=comp)
                    for comp, color in compound_to_color.items()]

    # Put legend in bottom right subplot (axes[2, 1])
    axes[2, 1].axis('off')  # turn off axis
    axes[2, 1].legend(handles=legend_elements, loc='center',
                    bbox_to_anchor=(0.5, 0.5), ncol=2, frameon=True, 
                    fontsize=10, title='Compounds')

    plt.suptitle('Model Feature Representations Across Processing Stages', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_features_umap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Model features UMAP plots saved")


def create_cross_modal_alignment_plot(feature_data, output_dir):
    """Create comprehensive cross-modal alignment analysis."""
    logger.info("Creating cross-modal alignment analysis...")
    
    # Extract and standardize features
    rna_features = StandardScaler().fit_transform(feature_data['rna_features'])
    image_features = StandardScaler().fit_transform(feature_data['image_features'])
    rna_enhanced = StandardScaler().fit_transform(feature_data['rna_enhanced'])
    image_enhanced = StandardScaler().fit_transform(feature_data['image_enhanced'])
    shared_representations = feature_data['shared_representations']
    
    n_samples = len(rna_features)
    
    # Calculate quantitative alignment metrics
    alignment_metrics = {}
    
    # 1. Feature correlation analysis (if dimensions allow)
    if rna_enhanced.shape[1] == image_enhanced.shape[1]:
        # Sample-wise correlations after enhancement
        enhanced_correlations = []
        for i in range(n_samples):
            corr, _ = pearsonr(rna_enhanced[i], image_enhanced[i])
            if not np.isnan(corr):
                enhanced_correlations.append(corr)
        
        alignment_metrics['enhanced_correlation'] = {
            'mean': float(np.mean(enhanced_correlations)),
            'std': float(np.std(enhanced_correlations)),
            'median': float(np.median(enhanced_correlations))
        }
        
        logger.info(f"Enhanced feature correlation: {alignment_metrics['enhanced_correlation']['mean']:.4f} ± {alignment_metrics['enhanced_correlation']['std']:.4f}")
    
    # 2. Shared space analysis
    from scipy.spatial.distance import pdist, squareform
    shared_distances = squareform(pdist(shared_representations, metric='cosine'))
    
    # Calculate treatment separation in shared space
    compounds = [label.split('_')[0] for label in feature_data['labels']]
    unique_compounds = list(set(compounds))

    intra_distances = []
    inter_distances = []
    
    for compound in unique_compounds:
        compound_indices = [i for i, c in enumerate(compounds) if c == compound]
        if len(compound_indices) > 1:
            # Intra-compound distances
            for i in range(len(compound_indices)):
                for j in range(i + 1, len(compound_indices)):
                    intra_distances.append(shared_distances[compound_indices[i], compound_indices[j]])
        
        # Inter-compound distances (sample a few to avoid explosion)
        other_indices = [i for i, c in enumerate(compounds) if c != compound]
        for i in compound_indices:
            sampled_others = np.random.choice(other_indices, min(5, len(other_indices)), replace=False)
            for j in sampled_others:
                inter_distances.append(shared_distances[i, j])
    
    if intra_distances and inter_distances:
        separation_ratio = np.mean(inter_distances) / np.mean(intra_distances)
        alignment_metrics['shared_space_separation'] = {
            'intra_mean': float(np.mean(intra_distances)),
            'inter_mean': float(np.mean(inter_distances)),
            'separation_ratio': float(separation_ratio)
        }
        logger.info(f"Shared space separation ratio: {separation_ratio:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=300)
    
    # Plot 1: Before cross-modal attention (separate spaces)
    logger.info("Computing UMAP for features before cross-modal attention...")
    
    reducer_rna_before = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    rna_embedding_before = reducer_rna_before.fit_transform(rna_features)

    reducer_image_before = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=43)
    image_embedding_before = reducer_image_before.fit_transform(image_features)
    
    axes[0, 0].scatter(rna_embedding_before[:, 0], rna_embedding_before[:, 1], 
                      c='blue', alpha=0.8, s=30, label='RNA Features', edgecolors='darkblue', linewidth=0.5)
    axes[0, 0].scatter(image_embedding_before[:, 0], image_embedding_before[:, 1], 
                      c='red', alpha=0.8, s=30, label='Image Features', edgecolors='darkred', linewidth=0.5)
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].set_title('Before Cross-Modal Attention\n(Separate Feature Spaces)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: After cross-modal attention
    logger.info("Computing UMAP for features after cross-modal attention...")
    
    if rna_enhanced.shape[1] == image_enhanced.shape[1]:
        # Joint UMAP since they should be in aligned space
        combined_after = np.vstack([rna_enhanced, image_enhanced])
        modality_labels = ['RNA'] * n_samples + ['Image'] * n_samples
        colors = ['blue' if label == 'RNA' else 'red' for label in modality_labels]
        edges = ['darkblue' if label == 'RNA' else 'darkred' for label in modality_labels]

        reducer_after = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
        embedding_after = reducer_after.fit_transform(combined_after)
        
        axes[0, 1].scatter(embedding_after[:, 0], embedding_after[:, 1], 
                          c=colors, alpha=0.8, s=30, edgecolors=edges, linewidth=0.5)
        
        # Add connecting lines between corresponding RNA and image samples
        for i in range(n_samples):
            axes[0, 1].plot([embedding_after[i, 0], embedding_after[i + n_samples, 0]], 
                           [embedding_after[i, 1], embedding_after[i + n_samples, 1]], 
                           'gray', alpha=0.5, linewidth=0.5)
        
        axes[0, 1].set_title('After Cross-Modal Attention\n(Joint Aligned Space)')
        
        # Add manual legend
        axes[0, 1].scatter([], [], c='blue', alpha=0.8, s=30, edgecolors='darkblue', 
                          linewidth=0.5, label='RNA Enhanced')
        axes[0, 1].scatter([], [], c='red', alpha=0.8, s=30, edgecolors='darkred', 
                          linewidth=0.5, label='Image Enhanced')
        axes[0, 1].legend()
    else:
        reducer_rna_after = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
        rna_embedding_after = reducer_rna_after.fit_transform(rna_enhanced)

        reducer_image_after = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=43)
        image_embedding_after = reducer_image_after.fit_transform(image_enhanced)
        
        axes[0, 1].scatter(rna_embedding_after[:, 0], rna_embedding_after[:, 1], 
                          c='blue', alpha=0.8, s=30, label='RNA Enhanced', edgecolors='darkblue', linewidth=0.5)
        axes[0, 1].scatter(image_embedding_after[:, 0], image_embedding_after[:, 1], 
                          c='red', alpha=0.8, s=30, label='Image Enhanced', edgecolors='darkred', linewidth=0.5)
        axes[0, 1].set_title('After Cross-Modal Attention\n(Enhanced but Separate Spaces)')
        axes[0, 1].legend()
    
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    if rna_enhanced.shape[1] == image_enhanced.shape[1]:
        # Use the same combined_after and embedding_after from Plot 2
        # Extract cell lines from labels
        cell_lines = [label.split('_',1)[1] for label in feature_data['labels']]  
        
        # Create cell line colors (using your existing mapping)
        cell_line_to_color = {
            'human_dermal_fibroblast': '#44AAFF', 
            'human_aortic_smooth_muscle_cell': '#32CD32', 
            'A549': '#FF4444'
        }
        
        # Create colors for both RNA and image samples (same cell line gets same color)
        cell_line_colors_rna = [cell_line_to_color[cl] for cl in cell_lines]
        cell_line_colors_image = [cell_line_to_color[cl] for cl in cell_lines]  # Same colors
        all_cell_line_colors = cell_line_colors_rna + cell_line_colors_image
        
        axes[0, 2].scatter(embedding_after[:, 0], embedding_after[:, 1], 
                        c=all_cell_line_colors, alpha=0.8, s=30, linewidth=0.5)
        
        # Add connecting lines between corresponding RNA and image samples
        for i in range(n_samples):
            axes[0, 2].plot([embedding_after[i, 0], embedding_after[i + n_samples, 0]], 
                        [embedding_after[i, 1], embedding_after[i + n_samples, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
        
        axes[0, 2].set_title('After Cross-Modal Attention\n(Colored by Cell Line)')
        
        # Add legend for cell lines
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=8, label=cell_line)
                        for cell_line, color in cell_line_to_color.items()]
        axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True)
        
    else:
        # Fallback for different dimensions (similar to Plot 2's else clause)
        reducer_rna_after = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
        rna_embedding_after = reducer_rna_after.fit_transform(rna_enhanced)
        
        reducer_image_after = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=43)
        image_embedding_after = reducer_image_after.fit_transform(image_enhanced)
        
        # Extract cell lines and create colors
        cell_lines = [label.split('_',1)[1] for label in feature_data['labels']]
        cell_line_to_color = {
            'human_dermal_fibroblast': '#44AAFF', 
            'human_aortic_smooth_muscle_cell': '#32CD32', 
            'A549': '#FF4444'
        }
        cell_line_sample_colors = [cell_line_to_color[cl] for cl in cell_lines]
        
        axes[0, 2].scatter(rna_embedding_after[:, 0], rna_embedding_after[:, 1], 
                        c=cell_line_sample_colors, alpha=0.8, s=30, 
                        label='RNA Enhanced', edgecolors='black', linewidth=0.5)
        axes[0, 2].scatter(image_embedding_after[:, 0], image_embedding_after[:, 1], 
                        c=cell_line_sample_colors, alpha=0.8, s=30, marker='^',
                        label='Image Enhanced', edgecolors='black', linewidth=0.5)
        axes[0, 2].set_title('After Cross-Modal Attention\n(Colored by Cell Line)')
        axes[0, 2].legend()

    axes[0, 2].set_xlabel('UMAP 1')
    axes[0, 2].set_ylabel('UMAP 2')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 3: Shared representations colored by compound
    logger.info("Computing UMAP for shared representations...")
    reducer_shared = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    shared_embedding = reducer_shared.fit_transform(shared_representations)
    logger.info(f"Number of samples in shared representations: {shared_representations.shape[0]}")
    
    # Get unique compounds for coloring
    unique_compounds = list(set(compounds))
    compound_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_compounds)))
    compound_to_color = dict(zip(unique_compounds, compound_colors))
    sample_colors = [compound_to_color[comp] for comp in compounds]
    
    axes[1, 0].scatter(shared_embedding[:, 0], shared_embedding[:, 1], 
                                c=sample_colors, alpha=0.8, s=30)
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].set_title('Shared Representations\n(Colored by Compound)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Shared representations colored by cell line
    cell_lines = [label.split('_',1)[1] for label in feature_data['labels']]  # Extract cell line from labels
    unique_cell_lines = list(set(cell_lines))
    
    # cmap = plt.cm.get_cmap('tab10')
    # tab10_colors = cmap.colors
    # rng = np.random.default_rng()
    # cell_line_colors = rng.choice(tab10_colors, size=len(unique_cell_lines), replace=False)
    # cell_line_to_color = dict(zip(unique_cell_lines, cell_line_colors))
    
    cell_line_to_color = {'human_dermal_fibroblast':'#44AAFF', 
                          'human_aortic_smooth_muscle_cell':'#32CD32', 
                          'A549':'#FF4444'}
    cell_line_sample_colors = [cell_line_to_color[cl] for cl in cell_lines]
    
    axes[1, 1].scatter(shared_embedding[:, 0], shared_embedding[:, 1],
                  c=cell_line_sample_colors, alpha=0.8, s=30)
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    axes[1, 1].set_title('Shared Representations\n(Colored by Cell Line)')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 4: Quantitative metrics summary  
    axes[1, 2].axis('off')
    
    # Create text summary of metrics
    metrics_text = "Cross-Modal Alignment Metrics\n" + "="*35 + "\n\n"
    
    if 'enhanced_correlation' in alignment_metrics:
        metrics_text += f"Enhanced Feature Correlation:\n"
        metrics_text += f"  Mean: {alignment_metrics['enhanced_correlation']['mean']:.4f}\n"
        metrics_text += f"  Std:  {alignment_metrics['enhanced_correlation']['std']:.4f}\n"
        metrics_text += f"  Median: {alignment_metrics['enhanced_correlation']['median']:.4f}\n\n"
    
    if 'shared_space_separation' in alignment_metrics:
        metrics_text += f"Shared Space Treatment Separation:\n"
        metrics_text += f"  Intra-compound distance: {alignment_metrics['shared_space_separation']['intra_mean']:.4f}\n"
        metrics_text += f"  Inter-compound distance: {alignment_metrics['shared_space_separation']['inter_mean']:.4f}\n"
        metrics_text += f"  Separation ratio: {alignment_metrics['shared_space_separation']['separation_ratio']:.4f}\n\n"
    
    metrics_text += f"Feature Dimensions:\n"
    metrics_text += f"  RNA features: {rna_features.shape[1]}\n"
    metrics_text += f"  Image features: {image_features.shape[1]}\n"
    metrics_text += f"  RNA enhanced: {rna_enhanced.shape[1]}\n"
    metrics_text += f"  Image enhanced: {image_enhanced.shape[1]}\n"
    metrics_text += f"  Shared repr: {shared_representations.shape[1]}\n"
    
    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes, 
                   verticalalignment='top', fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Cross-Modal Alignment Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'cross_modal_alignment.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save quantitative metrics to JSON
    with open(os.path.join(output_dir, 'cross_modal_alignment_metrics.json'), 'w') as f:
        json.dump(alignment_metrics, f, indent=2)
    
    logger.info("Cross-modal alignment analysis saved")

def save_generation_examples(generated_data, output_dir, num_examples=50):
    """Save visual examples of generated images and individual PNG files."""
    logger.info(f"Saving {num_examples} generation examples...")
   
    # Create subfolder for individual images
    individual_images_dir = os.path.join(output_dir, 'individual_images')
    os.makedirs(individual_images_dir, exist_ok=True)
   
    # Create comparison plot for first 10 examples only
    plot_examples = min(10, num_examples)
    fig, axes = plt.subplots(3, plot_examples, figsize=(3*plot_examples, 9), dpi=300)
    
    # Handle case where we might have fewer than 10 examples
    if plot_examples == 1:
        axes = axes.reshape(-1, 1)
   
    # Save all individual images but only plot first 10
    for i in range(num_examples):
        # Process control image
        control_img = generated_data['control_images'][i][:3].permute(1, 2, 0).numpy()
        control_img = (control_img + 1) / 2
        control_img = np.clip(control_img, 0, 1)
       
        # Process real treatment image
        real_img = generated_data['treatment_images_real'][i][:3].permute(1, 2, 0).numpy()
        real_img = (real_img + 1) / 2
        real_img = np.clip(real_img, 0, 1)
       
        # Process generated treatment image
        gen_img = generated_data['treatment_images_generated'][i][:3].permute(1, 2, 0).numpy()
        gen_img = (gen_img + 1) / 2
        gen_img = np.clip(gen_img, 0, 1)
       
        # Get metadata
        cell_line = generated_data['conditioning_info']['cell_line'][i]
        treatment = generated_data['conditioning_info']['treatment'][i]
        concentration = generated_data['conditioning_info']['compound_concentration_in_uM'][i]
       
        # Create safe filenames (remove special characters)
        safe_cell_line = "".join(c for c in cell_line if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_treatment = "".join(c for c in treatment if c.isalnum() or c in (' ', '-', '_')).rstrip()
       
        # Convert to 8-bit images for PNG saving
        control_img_8bit = (control_img * 255).astype(np.uint8)
        real_img_8bit = (real_img * 255).astype(np.uint8)
        gen_img_8bit = (gen_img * 255).astype(np.uint8)
       
        # Save individual images as high-quality PNGs (for ALL examples)
        from PIL import Image
       
        control_filename = f"sample_{i:03d}_control_{safe_cell_line}.png"
        Image.fromarray(control_img_8bit).save(
            os.path.join(individual_images_dir, control_filename),
            'PNG', compress_level=0
        )
       
        real_filename = f"sample_{i:03d}_real_{safe_treatment}_{concentration:.1f}uM_{safe_cell_line}.png"
        Image.fromarray(real_img_8bit).save(
            os.path.join(individual_images_dir, real_filename),
            'PNG', compress_level=0
        )
       
        gen_filename = f"sample_{i:03d}_generated_{safe_treatment}_{concentration:.1f}uM_{safe_cell_line}.png"
        Image.fromarray(gen_img_8bit).save(
            os.path.join(individual_images_dir, gen_filename),
            'PNG', compress_level=0
        )
        
        # Only plot the first 10 examples in the comparison figure
        if i < plot_examples:
            axes[0, i].imshow(control_img)
            axes[0, i].set_title(f"Control\n{cell_line}", fontsize=10)
            axes[0, i].axis('off')
           
            axes[1, i].imshow(real_img)
            axes[1, i].set_title(f"Real Treatment\n{treatment}", fontsize=10)
            axes[1, i].axis('off')
           
            axes[2, i].imshow(gen_img)
            axes[2, i].set_title(f"Generated\n{concentration:.1f}μM", fontsize=10)
            axes[2, i].axis('off')
   
    # Add row labels for clarity
    if plot_examples > 1:
        axes[0, 0].text(-0.1, 0.5, 'Control', rotation=90, transform=axes[0, 0].transAxes, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Real Treatment', rotation=90, transform=axes[1, 0].transAxes, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[2, 0].text(-0.1, 0.5, 'Generated', rotation=90, transform=axes[2, 0].transAxes, 
                       ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Save comparison plot (first 10 examples only)
    plt.suptitle(f'Drug Response Generation Examples (First {plot_examples} of {num_examples})', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_examples.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
   
    logger.info("Generation examples saved")
    logger.info(f"Comparison plot shows first {plot_examples} examples")
    logger.info(f"Individual images saved to {individual_images_dir}")
    logger.info(f"Saved {num_examples * 3} individual PNG files")

def main():
    parser = argparse.ArgumentParser(description="Generate results for different model types.")
    
    # Model and data paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to saved model checkpoint')
    parser.add_argument('--ablation', type=str, default='none', 
                       choices=['none', 'rna_only', 'image_only'],
                       help='Model type: none (full), rna_only, or image_only')
    
    parser.add_argument('--metadata_control', type=str, required=True,
                       help='Path to control metadata CSV')
    parser.add_argument('--metadata_drug', type=str, required=True,
                       help='Path to drug metadata CSV')
    parser.add_argument('--gene_count_matrix', type=str, required=True,
                       help='Path to gene count matrix parquet')
    parser.add_argument('--image_json_path', type=str, required=True,
                       help='Path to image paths JSON')
    parser.add_argument('--drug_data_path', type=str, required=True,
                       help='Path to drug data pickle file')
    
    # Output and generation parameters
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_generation_samples', type=int, default=200,
                       help='Number of samples to generate')
    parser.add_argument('--num_feature_samples', type=int, default=500,
                       help='Number of samples for feature extraction')
    parser.add_argument('--generation_steps', type=int, default=100,
                       help='Number of generation steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating {args.ablation} model")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, compound_to_idx, cell_line_to_idx = load_model_from_checkpoint(
        args.checkpoint_path, device, args.ablation
    )
    
    # Load data
    logger.info("Loading validation data...")
    metadata_control = pd.read_csv(args.metadata_control)
    metadata_drug = pd.read_csv(args.metadata_drug)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix)
    
    # Create validation DataLoader
    full_dataloader, _ = create_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        transform=image_transform,
        target_size=args.img_size,
        drug_data_path=args.drug_data_path,
    )
    
    # Create validation dataset (take last 20% as validation)
    dataset_size = len(full_dataloader.dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    _, val_dataset = torch.utils.data.random_split(
        full_dataloader.dataset, [train_size, val_size]
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=full_dataloader.collate_fn
    )
    
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Initialize rectified flow (needed for image generation)
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    
    # Model-specific evaluation
    if args.ablation == 'rna_only':
        logger.info("=== RNA-ONLY MODEL EVALUATION ===")
        
        # Generate RNA predictions
        rna_data = generate_rna_predictions(
            model, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_generation_samples, args.ablation
        )
        
        if rna_data:
            # Calculate and plot RNA metrics
            calculate_and_plot_rna_metrics(rna_data, args.output_dir)
            
            # Create RNA UMAP plots
            create_rna_umap_plot(rna_data, args.output_dir)
            
            logger.info("RNA-only model evaluation complete!")
        
    elif args.ablation == 'image_only':
        logger.info("=== IMAGE-ONLY MODEL EVALUATION ===")
        
        # Generate treatment images
        image_data = generate_treatment_images(
            model, rectified_flow, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_generation_samples, args.generation_steps, args.ablation
        )
        
        if image_data:
            # Save generation examples
            save_generation_examples(image_data, args.output_dir)
            
            logger.info("Image-only model evaluation complete!")
        
    else:  # Full cross-modal model
        logger.info("=== FULL CROSS-MODAL MODEL EVALUATION ===")
        
        # Generate RNA predictions
        rna_data = generate_rna_predictions(
            model, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_generation_samples, args.ablation
        )
        
        # Generate treatment images  
        image_data = generate_treatment_images(
            model, rectified_flow, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_generation_samples, args.generation_steps, args.ablation
        )
        
        # Extract model features for cross-modal analysis
        feature_data = extract_model_features(
            model, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_feature_samples, args.ablation
        )
        
        # RNA evaluation
        if rna_data:
            calculate_and_plot_rna_metrics(rna_data, args.output_dir)
            create_rna_umap_plot(rna_data, args.output_dir)
        
        # Image evaluation
        if image_data:
            save_generation_examples(image_data, args.output_dir)
        
        # Cross-modal feature analysis
        if feature_data:
            # Keep existing cross-modal analysis functions
            create_model_features_umap_plot(feature_data, args.output_dir)
            create_cross_modal_alignment_plot(feature_data, args.output_dir)
        
        logger.info("Full cross-modal model evaluation complete!")
    
    # Save summary statistics
    logger.info("Saving summary statistics...")
    summary_stats = {
        'model_type': args.ablation,
        'model_checkpoint': args.checkpoint_path,
        'generation_steps': args.generation_steps,
        'num_generation_samples': args.num_generation_samples
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()