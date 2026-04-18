import os
import sys
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics.aggregation import MeanMetric
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_main_process
from flow import MultiModalDOPRI5Solver
from evaluation import evaluate_drug_effects, evaluate_cross_modal_consistency, prepare_conditioning_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _process_training_batch(model, rectified_flow, batch, conditioning_tensors, 
                           device, use_amp, use_triplet, cross_modal_weights, ablation_type='none'):
    """Process a single training batch with ablation support."""
    control_rna = batch['control_transcriptomics'].to(device).float()
    control_images = batch['control_images'].to(device).float()
    treatment_rna = batch['treatment_transcriptomics'].to(device).float()
    treatment_images = batch['treatment_images'].to(device).float()
    
    with torch.amp.autocast('cuda', enabled=use_amp):
        if ablation_type == 'rna_only':
            # RNA-only model: only transcriptome prediction
            pred_treatment_rna = model(
                control_rna=control_rna,
                conditioning_info=conditioning_tensors,
                mode='transcriptome_direct'
            )

            mse = F.mse_loss(pred_treatment_rna, treatment_rna)

            # Compute Pearson correlation
            pred_centered = pred_treatment_rna - pred_treatment_rna.mean(dim=1, keepdim=True)
            target_centered = treatment_rna - treatment_rna.mean(dim=1, keepdim=True)
            correlation = F.cosine_similarity(pred_centered, target_centered, dim=1).mean()
            correlation = torch.clamp(correlation, -1.0, 1.0)
            pearson_loss = 1.0 - correlation
            
            # Assign transcriptome_loss
            transcriptome_loss = 0.9 * mse + 0.1 * pearson_loss
            image_loss = torch.tensor(0.0, device=device)  # No image loss
            total_loss = transcriptome_loss
            cross_modal_loss_dict = {'total_cross_modal': torch.tensor(0.0, device=device)}
            task_weights = [1.0, 0.0]
            
        elif ablation_type == 'image_only':
            # Image-only model: only image generation
            # Get shared representation for conditioning
            shared_repr = model(
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='shared'
            )['shared_representation']
            
            # Get image conditioning
            if hasattr(model, 'module'):
                image_conditioning = model.module.image_conditioning_head(shared_repr)
            else:
                image_conditioning = model.image_conditioning_head(shared_repr)
            
            # Image generation loss
            t = torch.rand(control_rna.shape[0], device=control_rna.device)
            path_sample = rectified_flow.sample_path(x_1=treatment_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]
            noise = path_sample["noise"]

            if hasattr(model, 'module'):
                v_pred = model.module.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})
            else:
                v_pred = model.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})

            if use_triplet:
                image_loss_dict = rectified_flow.loss_fn(
                    model_output=v_pred,
                    target_velocity=target_velocity,
                    target_images=treatment_images,
                    noise=noise,
                    labels=conditioning_tensors['compound_ids'],
                    use_triplet=True,
                    temperature=0.05
                )
                image_loss = image_loss_dict["loss"]
            else:
                image_loss = rectified_flow.loss_fn(v_pred, target_velocity)
            
            transcriptome_loss = torch.tensor(0.0, device=device)  # No RNA loss
            total_loss = image_loss
            cross_modal_loss_dict = {'total_cross_modal': torch.tensor(0.0, device=device)}
            task_weights = [0.0, 1.0]
            
        else:
            # Full cross-modal model — single forward pass through DDP wrapper
            t = torch.rand(control_rna.shape[0], device=device)
            path_sample = rectified_flow.sample_path(x_1=treatment_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]
            noise = path_sample["noise"]

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(
                    x=x_t, t=t,
                    control_rna=control_rna,
                    control_images=control_images,
                    conditioning_info=conditioning_tensors,
                    mode='train'
                )

            representations = outputs['representations']
            pred_treatment_rna = outputs['pred_rna']
            v_pred = outputs['v_pred']

            with torch.amp.autocast('cuda', enabled=use_amp):
                mse = F.mse_loss(pred_treatment_rna, treatment_rna)
                pred_centered = pred_treatment_rna - pred_treatment_rna.mean(dim=1, keepdim=True)
                target_centered = treatment_rna - treatment_rna.mean(dim=1, keepdim=True)
                correlation = torch.clamp(
                    F.cosine_similarity(pred_centered, target_centered, dim=1).mean(), -1.0, 1.0
                )
                transcriptome_loss = 0.9 * mse + 0.1 * (1.0 - correlation)

                if use_triplet:
                    image_loss_dict = rectified_flow.loss_fn(
                        model_output=v_pred,
                        target_velocity=target_velocity,
                        target_images=treatment_images,
                        noise=noise,
                        labels=conditioning_tensors['compound_ids'],
                        use_triplet=True,
                        temperature=0.05
                    )
                    image_loss = image_loss_dict["loss"]
                else:
                    image_loss = rectified_flow.loss_fn(v_pred, target_velocity)

                if hasattr(model, 'module'):
                    cross_modal_loss_dict = model.module.compute_cross_modal_consistency_loss(
                        representations, treatment_rna, treatment_images
                    )
                else:
                    cross_modal_loss_dict = model.compute_cross_modal_consistency_loss(
                        representations, treatment_rna, treatment_images
                    )

                total_cross_modal_loss = (
                    cross_modal_weights['alignment'] * cross_modal_loss_dict.get('cross_modal_alignment', 0.0)
                    + cross_modal_weights['variance'] * cross_modal_loss_dict.get('variance_regularization', 0.0)
                )
                cross_modal_loss_dict['total_cross_modal'] = total_cross_modal_loss

                total_loss = 0.50 * transcriptome_loss + 0.50 * image_loss + total_cross_modal_loss
                task_weights = [0.50, 0.50]
    
    return total_loss, transcriptome_loss.item(), image_loss.item(), task_weights, cross_modal_loss_dict

def _process_validation_batch(model, rectified_flow, batch, conditioning_tensors, device, use_amp, ablation_type='none'):
    """Process a single validation batch with ablation support."""
    control_rna = batch['control_transcriptomics'].to(device).float()
    control_images = batch['control_images'].to(device).float()
    treatment_rna = batch['treatment_transcriptomics'].to(device).float()
    treatment_images = batch['treatment_images'].to(device).float()
    
    with torch.amp.autocast('cuda', enabled=use_amp):
        if ablation_type == 'rna_only':
            # RNA-only validation
            pred_treatment_rna = model(
                control_rna=control_rna,
                conditioning_info=conditioning_tensors,
                mode='transcriptome_direct'
            )
            total_loss = F.mse_loss(pred_treatment_rna, treatment_rna)
            
        elif ablation_type == 'image_only':
            # Image-only validation
            shared_repr = model(
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='shared'
            )['shared_representation']
            
            if hasattr(model, 'module'):
                image_conditioning = model.module.image_conditioning_head(shared_repr)
            else:
                image_conditioning = model.image_conditioning_head(shared_repr)
            
            t = torch.rand(control_rna.shape[0], device=control_rna.device)
            path_sample = rectified_flow.sample_path(x_1=treatment_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]

            if hasattr(model, 'module'):
                v_pred = model.module.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})
            else:
                v_pred = model.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})

            total_loss = rectified_flow.loss_fn(v_pred, target_velocity)
            
        else:
            # Full cross-modal validation
            representations = model(
                x=None, t=None,
                control_rna=control_rna,
                control_images=control_images,
                conditioning_info=conditioning_tensors,
                mode='shared'
            )
            shared_repr = representations['shared_representation']
            
            # Compute transcriptome loss using shared representation
            if hasattr(model, 'module'):
                pred_treatment_rna = model.module.transcriptome_direct_head(shared_repr)
            else:
                pred_treatment_rna = model.transcriptome_direct_head(shared_repr)
            
            transcriptome_loss = F.mse_loss(pred_treatment_rna, treatment_rna)
            
            # Compute image loss using shared representation
            if hasattr(model, 'module'):
                image_conditioning = model.module.image_conditioning_head(shared_repr)
            else:
                image_conditioning = model.image_conditioning_head(shared_repr)
            
            t = torch.rand(control_rna.shape[0], device=control_rna.device)
            path_sample = rectified_flow.sample_path(x_1=treatment_images, t=t)
            x_t = path_sample["x_t"]
            target_velocity = path_sample["velocity"]

            if hasattr(model, 'module'):
                v_pred = model.module.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})
            else:
                v_pred = model.unet(x_t, t, extra={"multimodal_conditioning": image_conditioning})

            image_loss = rectified_flow.loss_fn(v_pred, target_velocity, use_triplet=False)
            total_loss = (transcriptome_loss + image_loss) / 2
    
    return total_loss

def _save_checkpoint(model, optimizer, compound_to_idx, cell_line_to_idx,
                     epoch, val_loss, best_model_path):
    actual_model = model.module if hasattr(model, 'module') else model

    model_config = {
        'model_type': actual_model.__class__.__name__,
        'compound_vocab_size': len(compound_to_idx),
        'cell_line_vocab_size': len(cell_line_to_idx),
    }

    for attr in ('rna_dim', 'img_channels', 'img_size', 'shared_embed_dim',
                 'model_channels', 'rna_output_dim', 'gene_embed_dim', 'image_output_dim'):
        if hasattr(actual_model, attr):
            model_config[attr] = getattr(actual_model, attr)

    if hasattr(actual_model, 'drug_embedding'):
        de = actual_model.drug_embedding
        model_config['drug_embed_dim'] = de.embed_dim
        model_config['use_kg_drug_encoder'] = de.kg_encoder is not None
    else:
        model_config['use_kg_drug_encoder'] = False

    rna_enc = getattr(actual_model, 'control_rna_encoder', None)
    kg_gene_enc = getattr(rna_enc, 'kg_gene_encoder', None) if rna_enc else None
    model_config['use_kg_gene_encoder'] = kg_gene_enc is not None

    checkpoint_data = {
        'model': actual_model.state_dict(),
        'config': model_config,
        'vocab_mappings': {
            'compound_to_idx': compound_to_idx,
            'cell_line_to_idx': cell_line_to_idx,
        },
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }

    # Save KG mappings (small dicts, not the large edge tensors which are in state_dict buffers)
    if model_config['use_kg_drug_encoder'] and actual_model.drug_embedding.drug_to_kg_mapping:
        checkpoint_data['drug_to_kg_mapping'] = actual_model.drug_embedding.drug_to_kg_mapping

    if model_config['use_kg_gene_encoder'] and kg_gene_enc is not None:
        checkpoint_data['gene_to_kg_mapping'] = kg_gene_enc.gene_to_kg_mapping
        if rna_enc.gene_names:
            checkpoint_data['gene_names'] = rna_enc.gene_names

    torch.save(checkpoint_data, best_model_path)

def train_multimodal_drug_conditioned_model(
    model, train_loader, val_loader, rectified_flow, device, compound_to_idx, cell_line_to_idx,
    num_epochs=30, lr=1e-4, best_model_path="best_multimodal_drug_conditioned_model.pt",
    patience=10, use_amp=True, weight_decay=0.0, use_triplet_loss=True, cross_modal_weights=None,
    ablation_type='none'):
    
    # Set default cross-modal weights
    if cross_modal_weights is None:
        cross_modal_weights = {
            'alignment': 0.1,
            'variance': 0.01,
        }
    
    model.to(device)
    
    # Include task balancer parameters in optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    train_loss_metric = MeanMetric().to(device)
    val_loss_metric = MeanMetric().to(device)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    transcriptome_losses, image_losses = [], []
    task_weights_history = []
    cross_modal_loss_history = []
    counter = 0
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
            
        # Training phase
        model.train()
        train_loss_metric.reset()
        epoch_transcriptome_losses = []
        epoch_image_losses = []
        epoch_weights = []
        epoch_cross_modal_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", disable=not is_main_process()):
            conditioning_tensors = prepare_conditioning_batch(
                batch['conditioning_info'], compound_to_idx, cell_line_to_idx, device,
                drug_embeddings=batch.get('drug_embeddings', None)
            )
            
            total_loss, transcriptome_loss_val, image_loss_val, current_weights, cross_modal_loss_dict = _process_training_batch(
                model, rectified_flow, batch, conditioning_tensors, device, use_amp, use_triplet_loss, cross_modal_weights, ablation_type)
            
            # Backpropagation
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            train_loss_metric.update(total_loss)
            
            epoch_transcriptome_losses.append(transcriptome_loss_val)
            epoch_image_losses.append(image_loss_val)
            epoch_weights.append(current_weights)
            
            # Track cross-modal losses
            if cross_modal_loss_dict:
                epoch_cross_modal_losses.append({
                    k: v.item() if torch.is_tensor(v) else v 
                    for k, v in cross_modal_loss_dict.items()
                })
        
        # Validation phase
        model.eval()
        val_loss_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", disable=not is_main_process()):
                conditioning_tensors = prepare_conditioning_batch(
                    batch['conditioning_info'], compound_to_idx, cell_line_to_idx, device
                )
                
                total_loss = _process_validation_batch(
                    model, rectified_flow, batch, conditioning_tensors, device, use_amp, ablation_type)

                val_loss_metric.update(total_loss)
        
        # Synchronize metrics across all processes
        train_loss = train_loss_metric.compute()
        val_loss = val_loss_metric.compute()
        
        # Convert to Python floats for logging
        train_loss = train_loss.item()
        val_loss = val_loss.item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        transcriptome_losses.append(np.mean(epoch_transcriptome_losses))
        image_losses.append(np.mean(epoch_image_losses))
        task_weights_history.append(np.mean(epoch_weights, axis=0))
        
        # Track cross-modal losses
        if epoch_cross_modal_losses:
            avg_cross_modal = {}
            for key in epoch_cross_modal_losses[0].keys():
                avg_cross_modal[key] = np.mean([loss_dict[key] for loss_dict in epoch_cross_modal_losses])
            cross_modal_loss_history.append(avg_cross_modal)
        else:
            cross_modal_loss_history.append({})
        
        lr_scheduler.step()
        
        # Logging (only on main process)
        if is_main_process():
            current_weights = task_weights_history[-1]
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"  Transcriptome: {transcriptome_losses[-1]:.4f} (weight: {current_weights[0] if len(current_weights) > 0 else 'N/A'})")
            logger.info(f"  Image: {image_losses[-1]:.4f} (weight: {current_weights[1] if len(current_weights) > 1 else 'N/A'})")
            
            # Log cross-modal losses if available
            if cross_modal_loss_history[-1]:
                cm_losses = cross_modal_loss_history[-1]
                logger.info(f"  Cross-modal - Alignment: {cm_losses.get('cross_modal_alignment', 0):.4f}, "
                           f"Variance: {cm_losses.get('variance_regularization', 0):.4f}")

        # Save best model and early stopping (only on main process)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model, optimizer, compound_to_idx, cell_line_to_idx, 
                epoch, val_loss, best_model_path
            )
            if is_main_process():
                logger.info(f"Model saved with validation loss: {val_loss:.4f}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if is_main_process():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Evaluate cross-modal consistency every 5 epochs (only on main process)
        if (epoch + 1) % 5 == 0 and is_main_process() and ablation_type == 'none':
            consistency_metrics = evaluate_cross_modal_consistency(
                model, val_loader, compound_to_idx, cell_line_to_idx, device
            )
            if consistency_metrics:  # Only log if metrics were computed (main process only)
                logger.info(f"Cross-modal consistency metrics:")
                logger.info(f"  Alignment Score: {consistency_metrics['cross_modal_alignment']:.4f}")
                logger.info(f"  Feature Correlation: {consistency_metrics['feature_correlation']:.4f}")
                logger.info(f"  Shared Variance: {consistency_metrics['shared_variance_explained']:.4f}")
                logger.info(f"  Consistency Score: {consistency_metrics['consistency_score']:.4f}")
    
    return train_losses, val_losses, transcriptome_losses, image_losses, task_weights_history, cross_modal_loss_history