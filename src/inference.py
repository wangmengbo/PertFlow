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
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2
from PIL import Image
import pandas as pd
from typing import Dict, List, Optional
import torch.nn.functional as F

# For advanced image metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import RectifiedFlow, MultiModalDOPRI5Solver
from model import MultiModalDrugConditionedModel, RNAOnlyDrugConditionedModel, ImageOnlyDrugConditionedModel
from dataloader import create_dataloader, image_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GeneImportanceAnalyzer:
    """Analyze gene importance for image generation during inference."""
    
    def __init__(self, model, rectified_flow, device):
        self.model = model
        self.rectified_flow = rectified_flow
        self.device = device
        self.gene_importance_data = {}
    
    def compute_gene_importance_for_image(self, 
                                        control_rna: torch.Tensor,
                                        control_images: torch.Tensor,
                                        conditioning_info: Dict[str, torch.Tensor],
                                        target_images: torch.Tensor,
                                        method: str = 'gradient') -> torch.Tensor:
        """Compute gene importance for a batch of samples."""
        control_rna = control_rna.detach().requires_grad_(True)
        self.model.zero_grad()
        
        representations = self.model(
            x=None, t=None,
            control_rna=control_rna,
            control_images=control_images,
            conditioning_info=conditioning_info,
            mode='shared'
        )
        
        shared_repr = representations['shared_representation']
        image_conditioning = self.model.image_conditioning_head(shared_repr)
        
        batch_size = control_rna.shape[0]
        t = torch.ones(batch_size, device=self.device) * 0.5
        
        path_sample = self.rectified_flow.sample_path(x_1=target_images, t=t)
        x_t = path_sample["x_t"]
        target_velocity = path_sample["velocity"]
        
        v_pred = self.model.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})
        
        image_loss = F.mse_loss(v_pred, target_velocity, reduction='sum')
        image_loss.backward()
        
        gene_importance = torch.abs(control_rna.grad)
        return gene_importance.detach()
    
    
    def save_gene_importance_matrix(self, output_path: str, gene_names: Optional[List[str]] = None):
        """Save gene importance matrix as CSV with proper formatting."""
        
        if not self.gene_importance_data:
            logger.warning("No gene importance data collected")
            return
        
        # Convert to DataFrame
        importance_matrix = torch.stack(list(self.gene_importance_data.values()), dim=1)  # [genes, samples]
        importance_df = pd.DataFrame(
            importance_matrix.cpu().numpy(),
            index=gene_names if gene_names else [f"Gene_{i}" for i in range(importance_matrix.shape[0])],
            columns=list(self.gene_importance_data.keys())
        )
        
        # Save main results
        importance_df.to_csv(output_path)
        logger.info(f"Gene importance matrix saved to {output_path}")
        logger.info(f"Matrix shape: {importance_df.shape[0]} genes × {importance_df.shape[1]} samples")
        
        # Save summary statistics
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary_df = pd.DataFrame({
            'gene': importance_df.index,
            'mean_importance': importance_df.mean(axis=1),
            'std_importance': importance_df.std(axis=1),
            'max_importance': importance_df.max(axis=1),
            'min_importance': importance_df.min(axis=1),
            'median_importance': importance_df.median(axis=1)
        }).sort_values('mean_importance', ascending=False)
        
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Gene importance summary saved to {summary_path}")
        
        # Save top genes report
        top_genes_path = output_path.replace('.csv', '_top_genes.txt')
        with open(top_genes_path, 'w') as f:
            f.write("TOP 50 MOST IMPORTANT GENES FOR IMAGE GENERATION\n")
            f.write("=" * 60 + "\n\n")
            for i, (gene, row) in enumerate(summary_df.head(50).iterrows()):
                f.write(f"{i+1:2d}. {gene:20s} | Mean: {row['mean_importance']:.6f} ± {row['std_importance']:.6f}\n")
        
        logger.info(f"Top genes report saved to {top_genes_path}")


def run_gene_importance_analysis(model, val_loader, compound_to_idx, cell_line_to_idx,
                               rectified_flow, device, num_samples, output_dir,
                               gene_names: Optional[List[str]] = None,
                               method: str = 'gradient'):
    """Run gene importance analysis for image generation."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    analyzer = GeneImportanceAnalyzer(model, rectified_flow, device)
    
    logger.info(f"Running gene importance analysis using {method} method...")
    logger.info(f"Target samples: {num_samples}")
    
    samples_processed = 0
    model.eval()
    
    for batch in tqdm(val_loader, desc="Gene Importance Analysis"):
        if samples_processed >= num_samples:
            break
            
        batch_size = len(batch['control_transcriptomics'])
        samples_to_take = min(batch_size, num_samples - samples_processed)
        
        # Process ENTIRE batch at once
        control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
        control_images = batch['control_images'][:samples_to_take].to(device)
        treatment_images = batch['treatment_images'][:samples_to_take].to(device)
        
        conditioning_info = batch['conditioning_info']
        conditioning_subset = {}

        if isinstance(conditioning_info, list):
            selected_conditioning = conditioning_info[:samples_to_take]
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
        
        # Compute importance for the entire batch
        batch_importances = analyzer.compute_gene_importance_for_image(
            control_rna=control_rna,
            control_images=control_images,
            conditioning_info=conditioning_tensors,
            target_images=treatment_images,
            method=method
        )
        
        # Map batch results back to individual sample IDs
        for i in range(samples_to_take):
            compound = conditioning_subset['treatment'][i]
            cell_line = conditioning_subset['cell_line'][i]
            concentration = conditioning_subset['compound_concentration_in_uM'][i]
            sample_id = f"{compound}_{cell_line}_{concentration}uM_sample{samples_processed + i}"
            
            analyzer.gene_importance_data[sample_id] = batch_importances[i]
            
        samples_processed += samples_to_take
        logger.info(f"Processed {samples_processed}/{num_samples} samples so far")
    
    logger.info(f"Saving results for {len(analyzer.gene_importance_data)} samples...")
    output_path = os.path.join(output_dir, f'gene_importance_for_image_{method}.csv')
    analyzer.save_gene_importance_matrix(output_path, gene_names)
    
    logger.info(f"Completed gene importance analysis for {len(analyzer.gene_importance_data)} samples")
    
    return analyzer.gene_importance_data


########################################################################################################

def prepare_conditioning_batch(conditioning_info: dict, 
                              compound_to_idx: dict, 
                              cell_line_to_idx: dict,
                              device: torch.device) -> dict:
    """Convert conditioning info to tensors."""
    batch_size = len(conditioning_info['treatment'])
    
    compound_ids = torch.tensor([
        compound_to_idx[comp] for comp in conditioning_info['treatment']
    ], device=device)
    
    cell_line_ids = torch.tensor([
        cell_line_to_idx[cl] for cl in conditioning_info['cell_line']
    ], device=device)
    
    concentrations = torch.as_tensor(
        conditioning_info['compound_concentration_in_uM'], 
        dtype=torch.float32, device=device
    )

    timepoints = torch.as_tensor(
        conditioning_info['timepoint'],
        dtype=torch.float32, device=device
    )
    
    return {
        'compound_ids': compound_ids,
        'cell_line_ids': cell_line_ids,
        'concentration': concentrations,
        'timepoint': timepoints
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, ablation_type: str = 'none'):
    """Load model from checkpoint with proper configuration based on ablation type."""
    logger.info(f"Loading {ablation_type} model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model configuration
    config = checkpoint['config']
    vocab_mappings = checkpoint['vocab_mappings']
    
    # Check if model was trained with KG by examining fusion network size
    model_state_dict = checkpoint['model']
    fusion_weight = model_state_dict.get('drug_embedding.fusion_net.0.weight', None)
    
    # Initialize KG if the model was trained with it
    kg_processor = None
    kg_data = None
    drug_to_kg_mapping = None
    gene_to_kg_mapping = None
    gene_names = None
    use_kg_drug = False
    use_kg_gene = False
    
    if fusion_weight is not None and fusion_weight.shape[1] > 1800:
        logger.info("Model was trained with KG enabled, loading KG data...")
        try:
            from dataloader import PrimeKGProcessor
            
            kg_processor = PrimeKGProcessor()
            kg_data = kg_processor.load_and_process()
            
            # Create dummy mappings (inference doesn't need real KG mappings)
            drug_to_kg_mapping = {}
            gene_to_kg_mapping = {}
            gene_names = [f"gene_{i}" for i in range(config['rna_dim'])]
            use_kg_drug = True
            use_kg_gene = True
            
            logger.info("KG data loaded for inference compatibility")
        except Exception as e:
            logger.warning(f"Failed to load KG data: {e}. Model may not load correctly.")
    
    # Common parameters for all models
    common_params = {
        'compound_vocab_size': config['compound_vocab_size'],
        'cell_line_vocab_size': config['cell_line_vocab_size'],
        'compound_to_idx': vocab_mappings['compound_to_idx'],
        'rna_dim': config['rna_dim'],
        'img_channels': config.get('img_channels', 4),
        'img_size': config.get('img_size', 256),
        'drug_embed_dim': config.get('drug_embed_dim', 32),
        'shared_embed_dim': config.get('shared_embed_dim', 512),
        'use_kg_drug_encoder': use_kg_drug,
        'use_kg_gene_encoder': use_kg_gene,
        'kg_processor': kg_processor,
        'kg_data': kg_data,
        'drug_to_kg_mapping': drug_to_kg_mapping,
        'gene_to_kg_mapping': gene_to_kg_mapping,
        'gene_names': gene_names,
    }
    
    # Initialize appropriate model based on ablation type
    if ablation_type == 'none':
        # Full cross-modal model
        model = MultiModalDrugConditionedModel(
            **common_params,
            model_channels=128,
            rna_output_dim=256,
            gene_embed_dim=64,
            image_output_dim=256,
            # UNet parameters
            num_res_blocks=2,
            attention_resolutions=[16],
            dropout=0.1,
            channel_mult=(1, 2, 2, 2),
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
        )
        logger.info("Loaded full cross-modal model")
        
    elif ablation_type == 'rna_only':
        # RNA-only model
        model = RNAOnlyDrugConditionedModel(
            **common_params,
            rna_output_dim=256,
            gene_embed_dim=64,
        )
        logger.info("Loaded RNA-only model")
        
    elif ablation_type == 'image_only':
        # Image-only model
        model = ImageOnlyDrugConditionedModel(
            **common_params,
            model_channels=128,
            image_output_dim=256,
            num_res_blocks=2,
            attention_resolutions=[16],
            dropout=0.1,
            channel_mult=(1, 2, 2, 2),
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
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


class RNAMetricsCalculator:
    """Calculate comprehensive RNA prediction metrics."""
    
    def __init__(self):
        self.metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'pearson_correlation': [],
            'spearman_correlation': [],
            'r2_score': [],
            'explained_variance': [],
            'direction_accuracy': [],
            'change_magnitude_correlation': []
        }
    
    def calculate_sample_metrics(self, predicted: np.ndarray, target: np.ndarray, 
                                control: np.ndarray = None):
        """Calculate metrics for a single sample."""
        # Basic metrics
        mse = mean_squared_error(target, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target, predicted)
        
        # Correlation metrics
        pearson_corr, _ = pearsonr(predicted.flatten(), target.flatten())
        spearman_corr, _ = spearmanr(predicted.flatten(), target.flatten())
        
        # R-squared and explained variance
        ss_res = np.sum((target - predicted) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        explained_var = 1 - np.var(target - predicted) / (np.var(target) + 1e-8)
        
        sample_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'r2_score': r2,
            'explained_variance': explained_var
        }
        
        # Drug effect specific metrics if control is provided
        if control is not None:
            predicted_change = predicted - control
            target_change = target - control
            
            # Direction accuracy
            pred_direction = np.sign(predicted_change)
            target_direction = np.sign(target_change)
            direction_accuracy = np.mean(pred_direction == target_direction)
            
            # Change magnitude correlation
            if np.std(target_change) > 1e-8:
                change_corr, _ = pearsonr(predicted_change, target_change)
            else:
                change_corr = 0.0
                
            sample_metrics.update({
                'direction_accuracy': direction_accuracy,
                'change_magnitude_correlation': change_corr
            })
        
        return sample_metrics
    
    def update(self, sample_metrics):
        """Update running metrics with sample metrics."""
        for key, value in sample_metrics.items():
            if key in self.metrics and not np.isnan(value):
                self.metrics[key].append(value)
    
    def get_summary(self):
        """Get summary statistics of all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {'count': 0}
        return summary


class ImageMetricsCalculator:
    """Calculate comprehensive image quality metrics."""
    
    def __init__(self, device='cuda'):
        self.metrics = {
            'ssim': [],
            'psnr': [],
            'mse': [],
            'mae': [],
            'pearson_correlation': [],
            'color_mse': [],
            'edge_similarity': [],
            'texture_similarity': []
        }
        
        # Initialize LPIPS if available
        if LPIPS_AVAILABLE:
            self.lpips_metric = lpips.LPIPS(net='alex').to(device)
            self.metrics['lpips'] = []
        else:
            self.lpips_metric = None
            logger.warning("LPIPS not available, skipping perceptual metrics")
        
        self.device = device
    
    def preprocess_image_for_metrics(self, image: np.ndarray):
        """Preprocess image for metric calculation."""
        # Convert from (C, H, W) to (H, W, C) and normalize to 0-1
        if image.ndim == 3 and image.shape[0] <= 4:  # Channel first
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize to 0-1 range
        image = (image + 1) / 2  # From [-1, 1] to [0, 1]
        image = np.clip(image, 0, 1)
        
        return image
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray):
        """Calculate SSIM between two images."""
        img1 = self.preprocess_image_for_metrics(img1)
        img2 = self.preprocess_image_for_metrics(img2)
        
        if img1.ndim == 3:  # Multi-channel
            return ssim(img1, img2, channel_axis=2, data_range=1.0)
        else:
            return ssim(img1, img2, data_range=1.0)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray):
        """Calculate PSNR between two images."""
        img1 = self.preprocess_image_for_metrics(img1)
        img2 = self.preprocess_image_for_metrics(img2)
        
        return psnr(img1, img2, data_range=1.0)
    
    def calculate_lpips(self, img1: torch.Tensor, img2: torch.Tensor):
        """Calculate LPIPS perceptual distance."""
        if self.lpips_metric is None:
            return None
        
        # Ensure images are in correct format for LPIPS (RGB channels only)
        if img1.shape[0] > 3:
            img1 = img1[:3]  # Take first 3 channels
        if img2.shape[0] > 3:
            img2 = img2[:3]
        
        # Add batch dimension if needed
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        with torch.no_grad():
            distance = self.lpips_metric(img1, img2)
        
        return distance.item()
    
    def calculate_color_distribution_similarity(self, img1: np.ndarray, img2: np.ndarray):
        """Calculate color distribution similarity for H&E images."""
        img1 = self.preprocess_image_for_metrics(img1)
        img2 = self.preprocess_image_for_metrics(img2)
        
        # Calculate histograms for each channel
        color_mse = 0
        for c in range(min(img1.shape[2], 3)):  # RGB channels only
            hist1, _ = np.histogram(img1[:, :, c], bins=50, range=(0, 1), density=True)
            hist2, _ = np.histogram(img2[:, :, c], bins=50, range=(0, 1), density=True)
            color_mse += mean_squared_error(hist1, hist2)
        
        return color_mse / min(img1.shape[2], 3)
    
    def calculate_edge_similarity(self, img1: np.ndarray, img2: np.ndarray):
        """Calculate edge preservation similarity."""
        img1 = self.preprocess_image_for_metrics(img1)
        img2 = self.preprocess_image_for_metrics(img2)
        
        # Convert to grayscale if multi-channel
        if img1.ndim == 3:
            img1_gray = np.mean(img1[:, :, :3], axis=2)
            img2_gray = np.mean(img2[:, :, :3], axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate edges using Sobel
        edges1 = cv2.Sobel(img1_gray.astype(np.float32), cv2.CV_64F, 1, 1, ksize=3)
        edges2 = cv2.Sobel(img2_gray.astype(np.float32), cv2.CV_64F, 1, 1, ksize=3)
        
        # Calculate correlation between edge maps
        edges1_flat = edges1.flatten()
        edges2_flat = edges2.flatten()
        
        if np.std(edges1_flat) > 1e-8 and np.std(edges2_flat) > 1e-8:
            edge_corr, _ = pearsonr(edges1_flat, edges2_flat)
        else:
            edge_corr = 0.0
            
        return edge_corr
    
    def calculate_texture_similarity(self, img1: np.ndarray, img2: np.ndarray):
        """Calculate texture similarity using GLCM features."""
        img1 = self.preprocess_image_for_metrics(img1)
        img2 = self.preprocess_image_for_metrics(img2)
        
        # Convert to grayscale and scale to 0-255 for GLCM
        if img1.ndim == 3:
            img1_gray = np.mean(img1[:, :, :3], axis=2)
            img2_gray = np.mean(img2[:, :, :3], axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        
        img1_gray = (img1_gray * 255).astype(np.uint8)
        img2_gray = (img2_gray * 255).astype(np.uint8)
        
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Calculate GLCM for both images
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm1 = graycomatrix(img1_gray, distances, angles, levels=256, symmetric=True, normed=True)
            glcm2 = graycomatrix(img2_gray, distances, angles, levels=256, symmetric=True, normed=True)
            
            # Calculate texture properties
            contrast1 = graycoprops(glcm1, 'contrast').mean()
            contrast2 = graycoprops(glcm2, 'contrast').mean()
            
            energy1 = graycoprops(glcm1, 'energy').mean()
            energy2 = graycoprops(glcm2, 'energy').mean()
            
            # Calculate similarity as negative of absolute difference
            texture_sim = 1.0 / (1.0 + abs(contrast1 - contrast2) + abs(energy1 - energy2))
            
        except ImportError:
            logger.warning("scikit-image not available for texture analysis")
            texture_sim = 0.0
        
        return texture_sim
    
    def calculate_sample_metrics(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculate all metrics for a single sample."""
        # Convert to numpy for some metrics
        pred_np = predicted.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Basic metrics
        mse = float(torch.mean((predicted - target) ** 2).item())
        mae = float(torch.mean(torch.abs(predicted - target)).item())
        
        # Flatten for correlation
        pred_flat = predicted.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        if np.std(pred_flat) > 1e-8 and np.std(target_flat) > 1e-8:
            pearson_corr, _ = pearsonr(pred_flat, target_flat)
        else:
            pearson_corr = 0.0
        
        # Image-specific metrics
        ssim_score = self.calculate_ssim(pred_np, target_np)
        psnr_score = self.calculate_psnr(pred_np, target_np)
        color_mse = self.calculate_color_distribution_similarity(pred_np, target_np)
        edge_sim = self.calculate_edge_similarity(pred_np, target_np)
        texture_sim = self.calculate_texture_similarity(pred_np, target_np)
        
        sample_metrics = {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'mse': mse,
            'mae': mae,
            'pearson_correlation': pearson_corr,
            'color_mse': color_mse,
            'edge_similarity': edge_sim,
            'texture_similarity': texture_sim
        }
        
        # LPIPS if available
        if self.lpips_metric is not None:
            lpips_score = self.calculate_lpips(predicted, target)
            if lpips_score is not None:
                sample_metrics['lpips'] = lpips_score
        
        return sample_metrics
    
    def update(self, sample_metrics):
        """Update running metrics with sample metrics."""
        for key, value in sample_metrics.items():
            if key in self.metrics and not np.isnan(value):
                self.metrics[key].append(value)
    
    def get_summary(self):
        """Get summary statistics of all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {'count': 0}
        return summary


def calculate_fid_score(real_images_dir: str, generated_images_dir: str, device: str):
    """Calculate FID score between real and generated images."""
    if not FID_AVAILABLE:
        logger.warning("pytorch-fid not available, skipping FID calculation")
        return None
    
    try:
        fid = fid_score.calculate_fid_given_paths(
            [real_images_dir, generated_images_dir],
            batch_size=50,
            device=device,
            dims=2048
        )
        return fid
    except Exception as e:
        logger.error(f"Error calculating FID: {e}")
        return None


def run_rna_inference(model, val_loader, compound_to_idx, cell_line_to_idx, 
                     device, num_samples, ablation_type):
    """Run RNA inference and calculate metrics."""
    logger.info(f"Running RNA inference for {num_samples} samples ({ablation_type} model)...")
    
    if ablation_type == 'image_only':
        logger.warning("Image-only model cannot generate RNA predictions!")
        return None
    
    rna_calculator = RNAMetricsCalculator()
    samples_processed = 0
    
    all_predictions = []
    all_targets = []
    all_controls = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="RNA Inference"):
            if samples_processed >= num_samples:
                break
            
            batch_size = len(batch['control_transcriptomics'])
            samples_to_take = min(batch_size, num_samples - samples_processed)
            
            # Extract batch data
            control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
            treatment_rna_real = batch['treatment_transcriptomics'][:samples_to_take]
            
            # Prepare conditioning - handle both list and dict formats like evaluation.py
            conditioning_info = batch['conditioning_info']
            conditioning_subset = {}

            if isinstance(conditioning_info, list):
                # Handle list format from dataloader
                selected_conditioning = conditioning_info[:samples_to_take]
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
            
            # Generate RNA predictions
            if ablation_type == 'rna_only':
                treatment_rna_pred = model(
                    control_rna=control_rna,
                    conditioning_info=conditioning_tensors,
                    mode='transcriptome_direct'
                )
            else:  # Full model
                control_images = batch['control_images'][:samples_to_take].to(device)
                treatment_rna_pred = model(
                    x=None, t=None,
                    control_rna=control_rna,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors,
                    mode='transcriptome_direct'
                )
            
            # Calculate metrics for each sample
            for i in range(samples_to_take):
                pred_sample = treatment_rna_pred[i].cpu().numpy()
                target_sample = treatment_rna_real[i].numpy()
                control_sample = control_rna[i].cpu().numpy()
                
                sample_metrics = rna_calculator.calculate_sample_metrics(
                    pred_sample, target_sample, control_sample
                )
                rna_calculator.update(sample_metrics)
                
                # Store for later analysis
                all_predictions.append(pred_sample)
                all_targets.append(target_sample)
                all_controls.append(control_sample)
            
            samples_processed += samples_to_take
    
    logger.info(f"Processed {samples_processed} RNA samples")
    
    return {
        'metrics': rna_calculator.get_summary(),
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'controls': np.array(all_controls)
    }


def run_image_inference(model, rectified_flow, val_loader, compound_to_idx, 
                       cell_line_to_idx, device, num_samples, generation_steps, 
                       ablation_type, save_images=False, output_dir=None):
    """Run image inference and calculate metrics."""
    if ablation_type == 'rna_only':
        logger.warning("RNA-only model cannot generate images!")
        return None
    
    logger.info(f"Running image inference for {num_samples} samples ({ablation_type} model)...")
    
    image_calculator = ImageMetricsCalculator(device)
    samples_processed = 0
    
    # Create directories for FID calculation if needed
    if save_images and output_dir:
        real_images_dir = os.path.join(output_dir, 'real_images')
        generated_images_dir = os.path.join(output_dir, 'generated_images')
        os.makedirs(real_images_dir, exist_ok=True)
        os.makedirs(generated_images_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Image Inference"):
            if samples_processed >= num_samples:
                break
            
            batch_size = len(batch['control_transcriptomics'])
            samples_to_take = min(batch_size, num_samples - samples_processed)
            
            # Extract batch data
            control_images = batch['control_images'][:samples_to_take].to(device)
            treatment_images_real = batch['treatment_images'][:samples_to_take]
            
            # Prepare conditioning - handle both list and dict formats like evaluation.py
            conditioning_info = batch['conditioning_info']
            conditioning_subset = {}

            if isinstance(conditioning_info, list):
                # Handle list format from dataloader
                selected_conditioning = conditioning_info[:samples_to_take]
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
            
            # Generate images
            if ablation_type == 'image_only':
                solver = MultiModalDOPRI5Solver(
                    model=model,
                    rectified_flow=rectified_flow,
                    control_rna=None,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors
                )
            else:  # Full model
                control_rna = batch['control_transcriptomics'][:samples_to_take].to(device)
                solver = MultiModalDOPRI5Solver(
                    model=model,
                    rectified_flow=rectified_flow,
                    control_rna=control_rna,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors
                )
            
            treatment_images_pred = solver.generate_sample(num_steps=generation_steps, device=device)
            treatment_images_pred = torch.clamp(treatment_images_pred, -1, 1)
            
            # Calculate metrics for each sample
            for i in range(samples_to_take):
                pred_sample = treatment_images_pred[i]
                target_sample = treatment_images_real[i].to(device)
                
                sample_metrics = image_calculator.calculate_sample_metrics(pred_sample, target_sample)
                image_calculator.update(sample_metrics)
                
                # Save images for FID calculation if requested
                if save_images and output_dir:
                    # Convert to PIL images and save
                    pred_img = pred_sample[:3].permute(1, 2, 0).cpu().numpy()
                    target_img = target_sample[:3].permute(1, 2, 0).cpu().numpy()

                    # Normalize to 0-255
                    pred_img = ((pred_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    target_img = ((target_img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    
                    # Save as PNG
                    Image.fromarray(pred_img).save(
                        os.path.join(generated_images_dir, f'generated_{samples_processed + i:04d}.png')
                    )
                    Image.fromarray(target_img).save(
                        os.path.join(real_images_dir, f'real_{samples_processed + i:04d}.png')
                    )
            
            samples_processed += samples_to_take
    
    logger.info(f"Processed {samples_processed} image samples")
    
    # Calculate FID if images were saved
    fid_score_value = None
    if save_images and output_dir:
        logger.info("Calculating FID score...")
        fid_score_value = calculate_fid_score(real_images_dir, generated_images_dir, str(device))
        if fid_score_value is not None:
            logger.info(f"FID Score: {fid_score_value:.4f}")
    
    return {
        'metrics': image_calculator.get_summary(),
        'fid_score': fid_score_value
    }


def save_results(rna_results, image_results, output_dir, ablation_type):
    """Save all results to files and create visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw metrics
    all_results = {
        'model_type': ablation_type,
        'rna_results': rna_results['metrics'] if rna_results else None,
        'image_results': image_results
    }
    
    with open(os.path.join(output_dir, 'inference_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary report
    report_lines = [
        f"INFERENCE RESULTS - {ablation_type.upper()} MODEL",
        "=" * 50,
        ""
    ]
    
    if rna_results:
        report_lines.extend([
            "RNA PREDICTION METRICS:",
            "-" * 25
        ])
        for metric, stats in rna_results['metrics'].items():
            if stats.get('count', 0) > 0:
                report_lines.append(
                    f"{metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(median: {stats['median']:.4f})"
                )
        report_lines.append("")
    
    if image_results:
        report_lines.extend([
            "IMAGE GENERATION METRICS:",
            "-" * 25
        ])
        for metric, stats in image_results['metrics'].items():
            if stats.get('count', 0) > 0:
                report_lines.append(
                    f"{metric:25s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(median: {stats['median']:.4f})"
                )
        
        if image_results.get('fid_score') is not None:
            report_lines.append(f"{'FID Score':25s}: {image_results['fid_score']:.4f}")
        report_lines.append("")
    
    # Save text report
    with open(os.path.join(output_dir, 'inference_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    for line in report_lines:
        print(line)
    
    logger.info(f"Results saved to {output_dir}")


def filter_dataset_by_drugs(dataset, include_only=None, exclude=None):
    """Filter dataset to include/exclude specific drugs."""
    if include_only is None and exclude is None:
        return dataset
    
    filtered_indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        drug_name = sample['conditioning_info']['treatment']
        
        # If include_only is specified, only keep those drugs
        if include_only is not None:
            if drug_name in include_only:
                filtered_indices.append(idx)
        # If exclude is specified, remove those drugs
        elif exclude is not None:
            if drug_name not in exclude:
                filtered_indices.append(idx)
        else:
            filtered_indices.append(idx)
    
    return torch.utils.data.Subset(dataset, filtered_indices)


def main():
    parser = argparse.ArgumentParser(description="Run inference evaluation on validation set.")
    
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
                       help='Path to drug data directory')

    # Inference parameters
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to evaluate')
    parser.add_argument('--generation_steps', type=int, default=50,
                       help='Number of generation steps for images')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')

    # Add these arguments after the existing parser arguments in main()
    parser.add_argument('--exclude_drugs_inference', type=str, nargs='*', default=['Dabrafenib'],
        help='List of drugs to exclude from inference validation set')
    parser.add_argument('--include_only_drugs', type=str, nargs='*', default=None,
        help='List of drugs to include ONLY in inference (for generalization testing)')
    parser.add_argument('--debug_mode', action='store_true', default=True,
                    help='Enable debug mode for faster loading')
    parser.add_argument('--debug_samples_inference', type=int, default=None,
                    help='Number of samples for inference in debug mode')
    
    # Additional options
    parser.add_argument('--save_images', action='store_true',
                       help='Save generated images for FID calculation')
    parser.add_argument('--evaluate_rna', action='store_true', default=True,
                       help='Evaluate RNA predictions')
    parser.add_argument('--evaluate_images', action='store_true', default=True,
                       help='Evaluate image generation')
    
    parser.add_argument('--analyze_gene_importance', action='store_true',
                      help='Analyze gene importance for image generation')
    parser.add_argument('--gene_names_file', type=str, default=None,
                      help='Path to file containing gene names (one per line)')
    parser.add_argument('--importance_method', type=str, default='gradient',
                      choices=['gradient', 'integrated_gradient', 'perturbation'],
                      help='Method for computing gene importance')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Running inference on {args.ablation} model")
    logger.info(f"Number of samples: {args.num_samples}")
    
    gene_names = None
    if args.gene_names_file and os.path.exists(args.gene_names_file):
        with open(args.gene_names_file, 'r') as f:
            gene_names = [line.strip() for line in f.readlines()]

    # Load model
    model, compound_to_idx, cell_line_to_idx = load_model_from_checkpoint(
        args.checkpoint_path, device, args.ablation
    )
    
    # Load data
    logger.info("Loading validation data...")
    metadata_control = pd.read_csv(args.metadata_control)
    metadata_drug = pd.read_csv(args.metadata_drug)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix)

    # Apply filtering BEFORE creating dataset
    if args.exclude_drugs_inference:
        logger.info(f"Excluding drugs from inference: {args.exclude_drugs_inference}")
        metadata_drug = metadata_drug[~metadata_drug['compound'].isin(args.exclude_drugs_inference)]

    if args.include_only_drugs:
        logger.info(f"Including only drugs: {args.include_only_drugs}")
        metadata_drug = metadata_drug[metadata_drug['compound'].isin(args.include_only_drugs)]
    
    # Create validation DataLoader
    full_dataloader, _ = create_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        transform=image_transform,
        target_size=args.img_size,
        drug_data_path=args.drug_data_path,
        debug_mode=args.debug_mode,
        debug_samples=args.debug_samples_inference,
    )
    
    # Create validation dataset (take last 20% as validation)
    dataset_size = len(full_dataloader.dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Replace the existing validation dataset creation with:
    _, val_dataset = torch.utils.data.random_split(
        full_dataloader.dataset, [train_size, val_size]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,  # CHANGED: from 4 to 0
        collate_fn=full_dataloader.collate_fn
    )

    logger.info(f"Filtered validation set size: {len(val_dataset)}")
    
    # Initialize rectified flow for image generation
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    
    # Run evaluations
    rna_results = None
    image_results = None
    
    if args.evaluate_rna and args.ablation != 'image_only':
        rna_results = run_rna_inference(
            model, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_samples, args.ablation
        )
    
    if args.evaluate_images and args.ablation != 'rna_only':
        image_results = run_image_inference(
            model, rectified_flow, val_loader, compound_to_idx, cell_line_to_idx,
            device, args.num_samples, args.generation_steps, args.ablation,
            args.save_images, args.output_dir
        )
        
    # Add gene importance analysis
    if args.analyze_gene_importance and args.ablation != 'rna_only':
        logger.info("Running gene importance analysis for image generation...")
        
        # Extract dynamic HVG gene names from the dataloader
        if gene_names is None and hasattr(full_dataloader.dataset, 'gene_count_matrix'):
            gene_names = full_dataloader.dataset.gene_count_matrix.index.tolist()
            
        gene_importance_data = run_gene_importance_analysis(
            model=model,  # Use 'model' not 'raw_model'
            val_loader=val_loader,
            compound_to_idx=compound_to_idx,
            cell_line_to_idx=cell_line_to_idx,
            rectified_flow=rectified_flow,
            device=device,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            gene_names=gene_names,
            method=args.importance_method
        )

    # Save results
    save_results(rna_results, image_results, args.output_dir, args.ablation)
    
    logger.info("Inference evaluation complete!")


if __name__ == "__main__":
    main()

