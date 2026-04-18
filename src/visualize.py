import os
import sys
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import RectifiedFlow, MultiModalDOPRI5Solver
from model import MultiModalDrugConditionedModel, ImageOnlyDrugConditionedModel
from dataloader import create_dataloader, image_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    """Load model from checkpoint."""
    logger.info(f"Loading {ablation_type} model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    vocab_mappings = checkpoint['vocab_mappings']
    
    common_params = {
        'compound_vocab_size': config['compound_vocab_size'],
        'cell_line_vocab_size': config['cell_line_vocab_size'],
        'rna_dim': config['rna_dim'],
        'img_channels': config.get('img_channels', 4),
        'img_size': config.get('img_size', 256),
        'drug_embed_dim': 32,
        'shared_embed_dim': 512,
    }
    
    if ablation_type == 'none':
        model = MultiModalDrugConditionedModel(
            **common_params,
            model_channels=128,
            rna_output_dim=256,
            gene_embed_dim=64,
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
    elif ablation_type == 'image_only':
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
    else:
        raise ValueError(f"Ablation type {ablation_type} not supported for image generation")
    
    # Filter out dynamic modules
    model_state_dict = checkpoint['model']
    filtered_state_dict = {}
    exclude_keys = ['rna_to_image_recon', 'image_to_rna_recon', 'mi_estimator']
    
    for key, value in model_state_dict.items():
        if not any(key.startswith(exclude_key) for exclude_key in exclude_keys):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, vocab_mappings['compound_to_idx'], vocab_mappings['cell_line_to_idx']

def collect_samples_by_cell_line_and_compound(val_loader, max_samples_per_group=1):
    """Collect one sample for each cell line + compound combination."""
    logger.info("Collecting samples by cell line and compound...")
    
    # Dictionary to store samples: {cell_line: {compound: sample_data}}
    samples_by_group = defaultdict(lambda: defaultdict(list))
    
    for batch in tqdm(val_loader, desc="Scanning data"):
        batch_size = len(batch['control_transcriptomics'])
        
        for i in range(batch_size):
            cell_line = batch['conditioning_info']['cell_line'][i]
            compound = batch['conditioning_info']['treatment'][i]
            concentration = batch['conditioning_info']['compound_concentration_in_uM'][i]
            
            # Only keep max_samples_per_group per combination
            if len(samples_by_group[cell_line][compound]) < max_samples_per_group:
                sample_data = {
                    'control_rna': batch['control_transcriptomics'][i],
                    'control_image': batch['control_images'][i],
                    'treatment_rna_real': batch['treatment_transcriptomics'][i],
                    'treatment_image_real': batch['treatment_images'][i],
                    'cell_line': cell_line,
                    'compound': compound,
                    'concentration': concentration.item() if hasattr(concentration, 'item') else concentration,
                    'timepoint': batch['conditioning_info']['timepoint'][i]
                }
                samples_by_group[cell_line][compound].append(sample_data)
    
    logger.info(f"Collected samples for {len(samples_by_group)} cell lines")
    for cell_line, compounds in samples_by_group.items():
        logger.info(f"  {cell_line}: {len(compounds)} compounds")
    
    return samples_by_group

def generate_treatment_images_for_samples(model, rectified_flow, samples_by_group, 
                                        compound_to_idx, cell_line_to_idx, device, 
                                        num_steps=100, ablation_type='none'):
    """Generate treatment images for collected samples."""
    logger.info("Generating treatment images for samples...")
    
    results = defaultdict(lambda: defaultdict(dict))
    
    with torch.no_grad():
        for cell_line, compounds in samples_by_group.items():
            logger.info(f"Processing cell line: {cell_line}")
            
            for compound, sample_list in compounds.items():
                for sample_idx, sample_data in enumerate(sample_list):
                    # Prepare single sample batch
                    control_image = sample_data['control_image'].unsqueeze(0).to(device)
                    
                    conditioning_info = {
                        'treatment': [sample_data['compound']],
                        'cell_line': [sample_data['cell_line']],
                        'compound_concentration_in_uM': [sample_data['concentration']],
                        'timepoint': [sample_data['timepoint']]
                    }
                    
                    conditioning_tensors = prepare_conditioning_batch(
                        conditioning_info, compound_to_idx, cell_line_to_idx, device
                    )
                    
                    # Generate based on model type
                    if ablation_type == 'image_only':
                        solver = MultiModalDOPRI5Solver(
                            model=model,
                            rectified_flow=rectified_flow,
                            control_rna=None,
                            control_images=control_image,
                            conditioning_info=conditioning_tensors
                        )
                    else:  # Full model
                        control_rna = sample_data['control_rna'].unsqueeze(0).to(device)
                        solver = MultiModalDOPRI5Solver(
                            model=model,
                            rectified_flow=rectified_flow,
                            control_rna=control_rna,
                            control_images=control_image,
                            conditioning_info=conditioning_tensors
                        )
                    
                    treatment_image_generated = solver.generate_sample(num_steps=num_steps, device=device)
                    treatment_image_generated = torch.clamp(treatment_image_generated, -1, 1)
                    
                    # Store results
                    results[cell_line][compound] = {
                        'control_image': sample_data['control_image'],
                        'treatment_image_real': sample_data['treatment_image_real'],
                        'treatment_image_generated': treatment_image_generated.squeeze(0).cpu(),
                        'compound': sample_data['compound'],
                        'concentration': sample_data['concentration'],
                        'cell_line': sample_data['cell_line']
                    }
    
    return results

def create_visualization_plots(results, output_dir, images_per_row=8):
    """Create visualization plots organized by cell line."""
    logger.info("Creating visualization plots...")
    
    def process_image(img_tensor):
        """Convert tensor to displayable image."""
        if img_tensor.dim() == 3:
            img = img_tensor[:3].permute(1, 2, 0).numpy()
        else:
            img = img_tensor.numpy()
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        return np.clip(img, 0, 1)
    
    for cell_line, compounds in results.items():
        logger.info(f"Creating plot for {cell_line}")
        
        # Get all compounds for this cell line
        compound_list = list(compounds.keys())
        n_compounds = len(compound_list)
        
        # Calculate figure dimensions
        n_rows = (n_compounds + images_per_row - 1) // images_per_row  # Ceiling division
        
        # Create figure with 3 rows per compound row (control, real, generated)
        fig, axes = plt.subplots(3 * n_rows, images_per_row, 
                               figsize=(2.5 * images_per_row, 3 * 3 * n_rows))
        
        # Handle case with single row/column
        if n_rows == 1 and images_per_row == 1:
            axes = axes.reshape(3, 1)
        elif n_rows == 1:
            axes = axes.reshape(3, -1)
        elif images_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        # Fill in the plots
        for idx, compound in enumerate(compound_list):
            row_group = idx // images_per_row
            col = idx % images_per_row
            
            sample_data = compounds[compound]
            
            # Process images
            control_img = process_image(sample_data['control_image'])
            real_img = process_image(sample_data['treatment_image_real'])
            generated_img = process_image(sample_data['treatment_image_generated'])
            
            # Plot control image
            control_row = row_group * 3
            axes[control_row, col].imshow(control_img)
            axes[control_row, col].set_title(f"Control\n{cell_line}", fontsize=10)
            axes[control_row, col].axis('off')
            
            # Plot real treatment image
            real_row = row_group * 3 + 1
            axes[real_row, col].imshow(real_img)
            axes[real_row, col].set_title(f"Real Treatment\n{compound}", fontsize=10)
            axes[real_row, col].axis('off')
            
            # Plot generated treatment image
            gen_row = row_group * 3 + 2
            axes[gen_row, col].imshow(generated_img)
            axes[gen_row, col].set_title(f"Generated\n{sample_data['concentration']:.1f}μM", fontsize=10)
            axes[gen_row, col].axis('off')
        
        # Hide unused subplots
        total_subplots = 3 * n_rows * images_per_row
        used_subplots = 3 * n_compounds
        
        if used_subplots < total_subplots:
            for idx in range(n_compounds, n_rows * images_per_row):
                row_group = idx // images_per_row
                col = idx % images_per_row
                
                for row_offset in range(3):
                    row = row_group * 3 + row_offset
                    axes[row, col].axis('off')
        
        # Add row labels on the left
        if n_compounds > 0:
            fig.text(0.02, 0.83, 'Control', rotation=90, ha='center', va='center', 
                    fontsize=12, fontweight='bold')
            fig.text(0.02, 0.5, 'Real Treatment', rotation=90, ha='center', va='center', 
                    fontsize=12, fontweight='bold')
            fig.text(0.02, 0.17, 'Generated', rotation=90, ha='center', va='center', 
                    fontsize=12, fontweight='bold')
        
        # Clean cell line name for filename
        safe_cell_line = "".join(c for c in cell_line if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
        
        plt.suptitle(f'Drug Response Generation - {cell_line}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, top=0.95)
        
        # Save plot
        filename = f'drug_response_visualization_{safe_cell_line}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate visualization images for each cell line and compound.")
    
    # Model and data paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to saved model checkpoint')
    parser.add_argument('--ablation', type=str, default='none', 
                       choices=['none', 'image_only'],
                       help='Model type: none (full) or image_only')
    
    parser.add_argument('--metadata_control', type=str, required=True,
                       help='Path to control metadata CSV')
    parser.add_argument('--metadata_drug', type=str, required=True,
                       help='Path to drug metadata CSV')
    parser.add_argument('--gene_count_matrix', type=str, required=True,
                       help='Path to gene count matrix parquet')
    parser.add_argument('--image_json_path', type=str, required=True,
                       help='Path to image paths JSON')
    
    # Output and generation parameters
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization plots')
    parser.add_argument('--generation_steps', type=int, default=100,
                       help='Number of generation steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--images_per_row', type=int, default=8,
                       help='Number of images per row in visualization')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Using {args.ablation} model")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, compound_to_idx, cell_line_to_idx = load_model_from_checkpoint(
        args.checkpoint_path, device, args.ablation
    )
    
    # Load data
    logger.info("Loading data...")
    metadata_control = pd.read_csv(args.metadata_control)
    metadata_drug = pd.read_csv(args.metadata_drug)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix)
    
    # Create DataLoader
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
    )
    
    # Create validation dataset (use last 20% as validation)
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
        num_workers=4
    )
    
    # Collect samples by cell line and compound
    samples_by_group = collect_samples_by_cell_line_and_compound(val_loader, max_samples_per_group=1)
    
    # Initialize rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    
    # Generate treatment images
    results = generate_treatment_images_for_samples(
        model, rectified_flow, samples_by_group, 
        compound_to_idx, cell_line_to_idx, device,
        args.generation_steps, args.ablation
    )
    
    # Create visualization plots
    create_visualization_plots(results, args.output_dir, args.images_per_row)
    
    logger.info(f"Visualization complete! Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()