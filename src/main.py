import os
import sys
import json
import torch
import logging
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow import RectifiedFlow
from model import MultiModalDrugConditionedModel, RNAOnlyDrugConditionedModel, ImageOnlyDrugConditionedModel
from dataloader import create_dataloader, image_transform
from utils import setup_distributed, cleanup_distributed, is_main_process, get_rank, unwrap_model, log_on_main
from train import train_multimodal_drug_conditioned_model
from evaluation import (evaluate_drug_effects, evaluate_cross_modal_consistency, 
                       evaluate_rna_only_performance, evaluate_image_only_performance, 
                       compare_ablation_results, generate_multimodal_drug_conditioned_outputs)

# Add these imports at the top
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def main():
    parser = argparse.ArgumentParser(description="Train drug-conditioned RNA to H&E image generator with Rectified Flow.")
    
    # Add distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Data paths
    parser.add_argument('--metadata_control', type=str, default='None', help='Path to control metadata CSV')
    parser.add_argument('--metadata_drug', type=str, default='None', help='Path to drug metadata CSV')
    parser.add_argument('--gene_count_matrix', type=str, default='None', help='Path to gene count matrix parquet')

    parser.add_argument('--drug_data_path', type=str, default='None', help='Path to drug data directory')
    parser.add_argument('--smiles_fallback_path', type=str, default=None, help='Path to CSV with compound_name,smiles for fallback')

    # Image paths
    parser.add_argument('--image_json_path', type=str, default='None', help='Path to image paths JSON')
    parser.add_argument('--output_dir', type=str, default='output/run', help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--img_channels', type=int, default=4, help='Number of image channels')
    parser.add_argument('--model_channels', type=int, default=128, help='Base model channels')
    parser.add_argument('--drug_embed_dim', type=int, default=256, help='Drug embedding dimension')
    parser.add_argument('--rna_output_dim', type=int, default=256, help='RNA encoder output dimension')
    parser.add_argument('--gene_embed_dim', type=int, default=64, help='Gene embedding dimension')
    parser.add_argument('--image_output_dim', type=int, default=256, help='Image encoder output dimension')
    parser.add_argument('--shared_embed_dim', type=int, default=512, help='Shared representation dimension')

    # KG parameters
    parser.add_argument('--use_kg_drug_encoder', action='store_true', default=True, help='Use KG for drug encoding')
    parser.add_argument('--use_kg_gene_encoder', action='store_true', default=True, help='Use KG for gene encoding')
    parser.add_argument('--kg_data_path', type=str, default='/depot/natallah/data/Mengbo/HnE_RNA/PertRF/drug/PrimeKG/PrimeKG.csv', help='Path to processed KG data')

    # Other parameters
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--gen_steps', type=int, default=100, help='Generation steps for DOPRI5 solver')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--only_inference', action='store_true', help='Skip training, only run inference')
    
    parser.add_argument('--ablation', type=str, default='none', 
                   choices=['none', 'rna_only', 'image_only'],
                   help='Ablation study type: none (full model), rna_only, or image_only')

    parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode with verbose logging')
    parser.add_argument('--debug_samples', type=int, default=None, help='Number of samples to use in debug mode')
    parser.add_argument('--debug_cell_lines', type=str, nargs='*', default=None, help='List of cell lines to use in debug mode')
    parser.add_argument('--debug_drugs', type=str, nargs='*', default=None, help='List of drugs to use in debug mode')
    parser.add_argument('--exclude_drugs', type=str, nargs='*', default=['Dabrafenib'], help='List of drugs to exclude from training/evaluation')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device based on local rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    log_on_main(f"Using device: {device}, Rank: {rank}/{world_size}")
    
    # Create output directory only on main process
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and seed
    set_seed(args.seed)
    
    # Load data (all processes load data)
    log_on_main("Loading data...")
    metadata_control = pd.read_csv(args.metadata_control)
    metadata_drug = pd.read_csv(args.metadata_drug)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix)
    
    log_on_main(f"Control samples: {len(metadata_control)}")
    log_on_main(f"Drug samples: {len(metadata_drug)}")
    log_on_main(f"Gene count matrix shape: {gene_count_matrix.shape}")
    
    # Create DataLoader with the new paired dataset
    log_on_main("Creating DataLoader...")

    fallback_smiles_dict = {}
    smiles_fallback_path = getattr(args, 'smiles_fallback_path', None)
    if smiles_fallback_path and os.path.exists(smiles_fallback_path):
        # Load from CSV: compound_name, smiles
        smiles_df = pd.read_csv(smiles_fallback_path)
        fallback_smiles_dict = dict(zip(smiles_df['compound_name'], smiles_df['smiles']))
        log_on_main(f"Loaded {len(fallback_smiles_dict)} SMILES for fallback")

    full_dataloader, (compound_to_idx, cell_line_to_idx) = create_dataloader(
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
        debug_mode=args.debug,
        debug_samples=args.debug_samples if hasattr(args, 'debug_samples') else None,
        debug_cell_lines=args.debug_cell_lines if hasattr(args, 'debug_cell_lines') else None,
        debug_drugs=args.debug_drugs if hasattr(args, 'debug_drugs') else None,
        exclude_drugs=args.exclude_drugs if hasattr(args, 'exclude_drugs') else None,
    )

    log_on_main(f"Compounds: {len(compound_to_idx)} unique")
    log_on_main(f"Cell lines: {len(cell_line_to_idx)} unique")
    log_on_main(f"Example compounds: {list(compound_to_idx.keys())[:5]}")
    
    # Split into train/val
    dataset_size = len(full_dataloader.dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataloader.dataset, [train_size, val_size], )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        collate_fn=full_dataloader.collate_fn,
        worker_init_fn=seed_worker, 
        pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=0, 
        collate_fn=full_dataloader.collate_fn,
        worker_init_fn=seed_worker, 
        pin_memory=True)
    
    log_on_main(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
    
    # Get gene dimension from the dataset
    sample_batch = next(iter(train_loader))
    gene_dim = sample_batch['control_transcriptomics'].shape[1]
    log_on_main(f"Gene dimension: {gene_dim}")
    
    # Initialize drug-conditioned model
    log_on_main(f"Initializing model with ablation: {args.ablation}")
    
    # Initialize KG if requested
    kg_processor = None
    kg_data = None
    drug_to_kg_mapping = None
    gene_to_kg_mapping = None
    gene_names = None

    if args.use_kg_drug_encoder or args.use_kg_gene_encoder:
        if args.kg_data_path is None:
            raise ValueError("KG data path required when using KG encoders")
        
        log_on_main("Loading knowledge graph data...")
        from dataloader import PrimeKGProcessor
        
        kg_processor = PrimeKGProcessor()
        kg_data = kg_processor.load_and_process()
        
        if args.use_kg_drug_encoder:
            # Get unique drug names from dataset
            unique_drugs = list(set(metadata_drug['compound'].unique()))
            drug_to_kg_mapping = kg_processor.get_drug_to_kg_mapping(unique_drugs)
            log_on_main(f"Mapped {len(drug_to_kg_mapping)} drugs to KG")
        
        if args.use_kg_gene_encoder:
            # Get gene names from dataset
            gene_names = gene_count_matrix.index.tolist()
            gene_to_kg_mapping = kg_processor.get_gene_to_kg_mapping(gene_names)
            log_on_main(f"Mapped {len(gene_to_kg_mapping)} genes to KG")

    # Common model parameters
    model_params = {
        'compound_vocab_size': len(compound_to_idx),
        'cell_line_vocab_size': len(cell_line_to_idx),
        'compound_to_idx': compound_to_idx,
        'img_channels': args.img_channels,
        'img_size': args.img_size,
        'model_channels': args.model_channels,
        'drug_embed_dim': args.drug_embed_dim,
        'shared_embed_dim': args.shared_embed_dim,
        'rna_output_dim': args.rna_output_dim,
        'gene_embed_dim': args.gene_embed_dim,
        'image_output_dim': args.image_output_dim,
        'rna_dim': gene_dim,
        'use_kg_drug_encoder': args.use_kg_drug_encoder,
        'use_kg_gene_encoder': args.use_kg_gene_encoder,
        'kg_processor': kg_processor,
        'kg_data': kg_data,
        'drug_to_kg_mapping': drug_to_kg_mapping,
        'gene_to_kg_mapping': gene_to_kg_mapping,
        'gene_names': gene_names[:gene_dim] if gene_names and len(gene_names) >= gene_dim else None,
        # UNet parameters
        'num_res_blocks': 2,
        'attention_resolutions': [16],
        'dropout': 0.1,
        'channel_mult': (1, 2, 2, 2),
        'use_checkpoint': False,
        'num_heads': 4,
        'num_head_channels': 32,
        'use_scale_shift_norm': True,
        'resblock_updown': True,
        'use_new_attention_order': True,
    }
    
    # Create appropriate model based on ablation
    if args.ablation == 'none':
        model = MultiModalDrugConditionedModel(**model_params)
        model_name = "full_multimodal"
    elif args.ablation == 'rna_only':
        model = RNAOnlyDrugConditionedModel(**model_params)
        model_name = "rna_only"
    elif args.ablation == 'image_only':
        model = ImageOnlyDrugConditionedModel(**model_params)
        model_name = "image_only"
    else:
        raise ValueError(f"Unknown ablation type: {args.ablation}")
    
    model.to(device)
    log_on_main(f"Model type: {model_name}")

    # Store reference to raw model before DDP wrapping
    raw_model = model
    
    # Wrap with DDP for training
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True
        )
        
    log_on_main(f"Model initialized with {sum(p.numel() for p in raw_model.parameters())} parameters")
    
    # Initialize rectified flow
    rectified_flow = RectifiedFlow(sigma_min=0.002, sigma_max=80.0)
    log_on_main("Initialized rectified flow")
    
    # Train model
    best_model_path = os.path.join(args.output_dir, f"best_{model_name}_model.pt")
    
    if not args.only_inference:
        log_on_main("Starting training...")
        train_losses, val_losses, transcriptome_losses, image_losses, _, cross_modal_loss_history = train_multimodal_drug_conditioned_model(
            model=model,  # Pass DDP-wrapped model for training
            train_loader=train_loader,
            val_loader=val_loader,
            rectified_flow=rectified_flow,
            device=device,
            compound_to_idx=compound_to_idx,
            cell_line_to_idx=cell_line_to_idx,
            num_epochs=args.epochs,
            lr=args.lr,
            best_model_path=best_model_path,
            patience=args.patience,
            use_amp=args.use_amp,
            weight_decay=args.weight_decay,
            ablation_type=args.ablation)
        
        log_on_main(f"Training complete. Best model saved at {best_model_path}")
        
        # Enhanced plotting only on main process
        if is_main_process():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Total losses
            axes[0,0].plot(train_losses, label='Train Loss')
            axes[0,0].plot(val_losses, label='Validation Loss')
            axes[0,0].set_title('Total Loss')
            axes[0,0].legend()

            # Individual task losses
            axes[0,1].plot(transcriptome_losses, label='Transcriptome')
            axes[0,1].plot(image_losses, label='Image')
            axes[0,1].set_title('Task-Specific Losses')
            axes[0,1].legend()

            # Cross-modal loss
            total_cross_modal = [loss_dict.get('total_cross_modal', 0.0) for loss_dict in cross_modal_loss_history]
            axes[1,0].plot(total_cross_modal, label='Cross-Modal Loss')
            axes[1,0].set_title('Cross-Modal Loss Over Time')
            axes[1,0].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "multitask_training_curves.png"))
        
    else:
        log_on_main(f"Skipping training. Loading existing model from {best_model_path}")
    
    # Synchronize all processes before evaluation
    if dist.is_initialized():
        dist.barrier()
    
    # Load best model for evaluation - SIMPLIFIED
    log_on_main("Loading best model for evaluation...")
    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
    raw_model.load_state_dict(checkpoint["model"])  # Load into raw model
    raw_model.eval()

    # Also load vocabulary mappings if they were saved
    if 'vocab_mappings' in checkpoint:
        compound_to_idx = checkpoint['vocab_mappings']['compound_to_idx']
        cell_line_to_idx = checkpoint['vocab_mappings']['cell_line_to_idx']
    
    # All evaluation uses raw_model (unwrapped)
    # Ablation-aware evaluation
    if args.ablation == 'none':
        # Full model - evaluate everything
        if is_main_process():
            log_on_main("Evaluating drug effects...")
            evaluate_drug_effects(
                model=raw_model,
                rectified_flow=rectified_flow,
                val_loader=val_loader,
                compound_to_idx=compound_to_idx,
                cell_line_to_idx=cell_line_to_idx,
                device=device,
                output_dir=args.output_dir,
                num_samples=8)
        
            # Generate examples only on main process
            log_on_main("Generating example drug effects...")
            
            # Get a validation batch
            val_batch = next(iter(val_loader))
            
            # Select first sample for detailed analysis
            control_rna = val_batch['control_transcriptomics'][:1].to(device)
            
            # Extract conditioning info properly (handling list format)
            conditioning_info = val_batch['conditioning_info']
            if isinstance(conditioning_info, list):
                first_condition = conditioning_info[0]
                cell_line = first_condition['cell_line']
                treatment = first_condition['treatment']
            else:
                cell_line = conditioning_info['cell_line'][0]
                treatment = conditioning_info['treatment'][0]

            # Create different drug conditions for comparison
            example_conditions = [
                {
                    'treatment': ['DMSO'],
                    'cell_line': [cell_line],
                    'timepoint': [24.0],
                    'compound_concentration_in_uM': [0.0],
                },
                {
                    'treatment': [treatment],
                    'cell_line': [cell_line],
                    'timepoint': [24.0],
                    'compound_concentration_in_uM': [1.0],
                },
                {
                    'treatment': [treatment], 
                    'cell_line': [cell_line],
                    'timepoint': [24.0],
                    'compound_concentration_in_uM': [10.0],
                }
            ]
            
            generated_examples = []
            with torch.no_grad():
                for condition in example_conditions:
                    try:
                        treatment_rna, treatment_images = generate_multimodal_drug_conditioned_outputs(
                            model=raw_model,  # Use raw model
                            rectified_flow=rectified_flow,
                            control_rna=control_rna,
                            control_images=val_batch['control_images'][:1].to(device),
                            conditioning_info=condition,
                            compound_to_idx=compound_to_idx,
                            cell_line_to_idx=cell_line_to_idx,
                            device=device,
                            num_steps=args.gen_steps)
                        generated_examples.append(treatment_images[0])
                        logger.info(f"Generated image for condition: {condition}")
                    except Exception as e:
                        logger.warning(f"Failed to generate for condition {condition}: {e}")
                        generated_examples.append(torch.zeros(args.img_channels, args.img_size, args.img_size))
            
            # Visualize dose-response
            if generated_examples:
                fig, axes = plt.subplots(1, len(generated_examples), figsize=(5*len(generated_examples), 5))
                
                titles = ['DMSO (Control)', 'Low Dose (1μM)', 'High Dose (10μM)']
                
                for i, (example, title) in enumerate(zip(generated_examples, titles)):
                    rgb_img = example[:3].permute(1, 2, 0).cpu().numpy()
                    rgb_img = (rgb_img + 1) / 2
                    rgb_img = np.clip(rgb_img, 0, 1)
                    
                    axes[i].imshow(rgb_img)
                    axes[i].set_title(title)
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "dose_response_example.png"), dpi=300, bbox_inches='tight')
                plt.close()

            log_on_main("Running comprehensive cross-modal consistency evaluation...")
            final_consistency_metrics = evaluate_cross_modal_consistency(
                raw_model, val_loader, compound_to_idx, cell_line_to_idx, device
            )
            
            if final_consistency_metrics:  # Only process if metrics were computed (main process only)
                logger.info("=== Final Cross-Modal Consistency Results ===")
                logger.info(f"Cross-Modal Alignment Score: {final_consistency_metrics['cross_modal_alignment']:.4f}")
                logger.info(f"RNA Prediction MSE: {final_consistency_metrics['rna_prediction_mse']:.4f}")
                logger.info(f"Feature Correlation: {final_consistency_metrics['feature_correlation']:.4f}")
                logger.info(f"Shared Variance Explained: {final_consistency_metrics['shared_variance_explained']:.4f}")
                logger.info(f"Consistency Score: {final_consistency_metrics['consistency_score']:.4f}")
                
                # Save metrics to file
                import json
                metrics_path = os.path.join(args.output_dir, "cross_modal_consistency_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(final_consistency_metrics, f, indent=2)
                logger.info(f"Consistency metrics saved to {metrics_path}")
                
    elif args.ablation == 'rna_only':
        # RNA-only model - evaluate RNA prediction only
        if is_main_process():
            log_on_main("Evaluating RNA-only model performance...")
            evaluate_rna_only_performance(
                model=raw_model,
                val_loader=val_loader, 
                compound_to_idx=compound_to_idx,
                cell_line_to_idx=cell_line_to_idx,
                device=device,
                output_dir=args.output_dir)
                
    elif args.ablation == 'image_only':
        # Image-only model - evaluate image generation only  
        if is_main_process():
            log_on_main("Evaluating Image-only model performance...")
            evaluate_image_only_performance(
                model=raw_model,
                rectified_flow=rectified_flow,
                val_loader=val_loader,
                compound_to_idx=compound_to_idx,
                cell_line_to_idx=cell_line_to_idx,
                device=device,
                output_dir=args.output_dir,
                num_steps=args.gen_steps)
    
    logger.info(f"All results saved to {args.output_dir}")

    # Cleanup distributed training
    cleanup_distributed()

if __name__ == "__main__":
    main()