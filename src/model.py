import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unet import MultiModalConditionedUNet
from encoders import RNAEncoder, ImageEncoder, DrugEmbedding, ResidualBlock

class CrossModalAttention(nn.Module):
    """
    Strong cross-modal attention with contrastive learning for RNA-image alignment.
    Forces meaningful cross-modal representations through multi-token attention and contrastive loss.
    """
    def __init__(self, rna_dim, image_dim, hidden_dim=256, num_heads=8, dropout=0.1,
                 aligned_dim=None, num_tokens=16, contrastive_temperature=0.07,
                 drug_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_tokens = num_tokens
        self.contrastive_temperature = contrastive_temperature
        self.aligned_dim = aligned_dim or max(rna_dim, image_dim)

        self.rna_proj = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim * num_tokens),
            nn.LayerNorm(hidden_dim * num_tokens),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * num_tokens),
            nn.LayerNorm(hidden_dim * num_tokens),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Drug context token — projected into the same token space so both
        # modalities can attend to the drug perturbation during cross-modal attention
        if drug_dim is not None:
            self.drug_token_proj = nn.Linear(drug_dim, hidden_dim)
        else:
            self.drug_token_proj = None

        self.rna_to_image_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.image_to_rna_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rna_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.image_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.rna_pool_attn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.image_pool_attn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())

        self.rna_out_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.aligned_dim),
            nn.LayerNorm(self.aligned_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_out_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.aligned_dim),
            nn.LayerNorm(self.aligned_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.rna_contrastive_head = nn.Sequential(
            nn.Linear(self.aligned_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.image_contrastive_head = nn.Sequential(
            nn.Linear(self.aligned_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)
        
    def _pool_tokens(self, tokens, pool_attn):
        """Pool multi-token representations using attention."""
        # tokens: [B, num_tokens, hidden_dim]
        attn_weights = pool_attn(tokens)  # [B, num_tokens, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (tokens * attn_weights).sum(dim=1)  # [B, hidden_dim]
        return pooled, attn_weights
        
    def compute_contrastive_alignment_loss(self, rna_features, image_features):
        batch_size = rna_features.shape[0]
        device = rna_features.device

        if batch_size <= 1:
            return {
                'contrastive_loss': torch.zeros([], device=device),
                'positive_similarity': torch.zeros([], device=device),
                'temperature': self.contrastive_temperature
            }

        rna_contrastive = F.normalize(self.rna_contrastive_head(rna_features), p=2, dim=1)
        image_contrastive = F.normalize(self.image_contrastive_head(image_features), p=2, dim=1)

        sim_matrix = torch.mm(rna_contrastive, image_contrastive.t()) / self.contrastive_temperature

        labels = torch.arange(batch_size, device=device)
        rna_to_image_loss = F.cross_entropy(sim_matrix, labels)
        image_to_rna_loss = F.cross_entropy(sim_matrix.t(), labels)
        contrastive_loss = (rna_to_image_loss + image_to_rna_loss) / 2

        return {
            'contrastive_loss': contrastive_loss,
            'positive_similarity': torch.diag(sim_matrix).mean(),
            'temperature': self.contrastive_temperature
        }
        
    def forward(self, rna_features, image_features, drug_features=None):
        batch_size = rna_features.shape[0]

        rna_tokens = self.rna_proj(rna_features).view(batch_size, self.num_tokens, self.hidden_dim)
        image_tokens = self.image_proj(image_features).view(batch_size, self.num_tokens, self.hidden_dim)

        # Append drug context as an extra token to both modality sequences.
        # This lets RNA and image features attend to the drug perturbation,
        # ensuring the cross-modal interaction is conditioned on the drug.
        if self.drug_token_proj is not None and drug_features is not None:
            drug_token = self.drug_token_proj(drug_features).unsqueeze(1)  # [B, 1, hidden_dim]
            rna_tokens = torch.cat([rna_tokens, drug_token], dim=1)    # [B, num_tokens+1, H]
            image_tokens = torch.cat([image_tokens, drug_token], dim=1) # [B, num_tokens+1, H]

        rna_self_attended, _ = self.rna_self_attn(rna_tokens, rna_tokens, rna_tokens)
        image_self_attended, _ = self.image_self_attn(image_tokens, image_tokens, image_tokens)

        rna_cross_attended, rna_cross_weights = self.rna_to_image_attn(
            query=rna_self_attended,
            key=image_self_attended,
            value=image_self_attended
        )
        image_cross_attended, image_cross_weights = self.image_to_rna_attn(
            query=image_self_attended,
            key=rna_self_attended,
            value=rna_self_attended
        )

        rna_enhanced_tokens = rna_self_attended + rna_cross_attended
        image_enhanced_tokens = image_self_attended + image_cross_attended

        rna_pooled, rna_pool_weights = self._pool_tokens(rna_enhanced_tokens, self.rna_pool_attn)
        image_pooled, image_pool_weights = self._pool_tokens(image_enhanced_tokens, self.image_pool_attn)

        rna_enhanced = self.dropout(self.rna_out_proj(rna_pooled))
        image_enhanced = self.dropout(self.image_out_proj(image_pooled))

        if rna_features.shape[1] == self.aligned_dim:
            rna_enhanced = rna_enhanced + rna_features
        if image_features.shape[1] == self.aligned_dim:
            image_enhanced = image_enhanced + image_features

        contrastive_info = self.compute_contrastive_alignment_loss(rna_enhanced, image_enhanced)

        return rna_enhanced, image_enhanced, {
            'rna_cross_weights': rna_cross_weights,
            'image_cross_weights': image_cross_weights,
            'rna_pool_weights': rna_pool_weights,
            'image_pool_weights': image_pool_weights,
            'contrastive_info': contrastive_info
        }

class MultiModalDrugConditionedModel(nn.Module):
    """
    Multi-modal model with shared representation architecture.
    Control RNA + Control Image + Drug → Shared Representation → [Treatment RNA, Treatment Image]
    """
    def __init__(self,
                rna_dim: int,
                compound_vocab_size: int,
                cell_line_vocab_size: int,
                compound_to_idx: Optional[Dict] = None,
                img_channels: int = 4,
                img_size: int = 256,
                model_channels: int = 128,
                drug_embed_dim: int = 32,
                shared_embed_dim: int = 512,
                rna_output_dim: int = 256,
                gene_embed_dim: int = 64,
                image_output_dim: int = 256,
                rna_relation_rank: int = 25,
                rna_num_heads: int = 4,
                cross_modal_heads: int = 8,
                cross_modal_aligned_dim: Optional[int] = None,
                use_kg_drug_encoder: bool = False,
                use_kg_gene_encoder: bool = False,
                kg_processor=None,
                kg_data=None,
                drug_to_kg_mapping=None,
                gene_to_kg_mapping=None,
                gene_names=None,
                **unet_kwargs):
        super().__init__()
        
        self.rna_dim = rna_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.shared_embed_dim = shared_embed_dim
        
        # Drug embedding with optional KG
        self.drug_embedding = DrugEmbedding(
            compound_vocab_size=compound_vocab_size,
            cell_line_vocab_size=cell_line_vocab_size,
            embed_dim=drug_embed_dim,
            use_kg=use_kg_drug_encoder,
            kg_processor=kg_processor,
            kg_data=kg_data,
            drug_to_kg_mapping=drug_to_kg_mapping,
            compound_to_idx=compound_to_idx
        )
        
        # Individual modality encoders
        self.control_rna_encoder = RNAEncoder(
            input_dim=rna_dim,
            hidden_dims=[512, 256],
            output_dim=rna_output_dim,
            dropout=0.1,
            use_gene_relations=False,
            num_heads=rna_num_heads,
            relation_rank=rna_relation_rank,
            gene_embed_dim=gene_embed_dim,
            use_kg=use_kg_gene_encoder,
            kg_processor=kg_processor,
            kg_data=kg_data,
            gene_to_kg_mapping=gene_to_kg_mapping,
            gene_names=gene_names[:rna_dim] if gene_names and len(gene_names) >= rna_dim else None  # FIXED: only pass the subset
        )

        self.control_image_encoder = ImageEncoder(img_channels, image_output_dim)

        if cross_modal_aligned_dim is None:
            cross_modal_aligned_dim = max(rna_output_dim, image_output_dim)

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            rna_dim=rna_output_dim,
            image_dim=image_output_dim, 
            hidden_dim=256,
            num_heads=cross_modal_heads,
            aligned_dim=cross_modal_aligned_dim,
            drug_dim=drug_embed_dim
        )
        
        # Shared encoder that combines all modalities
        shared_input_dim = cross_modal_aligned_dim*2 + drug_embed_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_input_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            
            nn.LayerNorm(shared_embed_dim)
        )
        
        self.transcriptome_direct_head = nn.Sequential(
            nn.Linear(shared_embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(1024, 512, dropout=0.1),
            nn.Linear(512, rna_dim)
        )
        self.image_conditioning_head = self._build_image_conditioning_head(shared_embed_dim, shared_embed_dim)
        
        unet_specific_params = {
            'num_res_blocks': unet_kwargs.get('num_res_blocks', 2),
            'attention_resolutions': unet_kwargs.get('attention_resolutions', [16]),
            'dropout': unet_kwargs.get('dropout', 0.1),
            'channel_mult': unet_kwargs.get('channel_mult', (1, 2, 2, 2)),
            'use_checkpoint': unet_kwargs.get('use_checkpoint', False),
            'num_heads': unet_kwargs.get('num_heads', 4),
            'num_head_channels': unet_kwargs.get('num_head_channels', 32),
            'use_scale_shift_norm': unet_kwargs.get('use_scale_shift_norm', True),
            'resblock_updown': unet_kwargs.get('resblock_updown', True),
            'use_new_attention_order': unet_kwargs.get('use_new_attention_order', True),
        }
        
        # UNet for image generation
        self.unet = MultiModalConditionedUNet(
            in_channels=img_channels,
            model_channels=model_channels, 
            out_channels=img_channels,
            multimodal_embed_dim=shared_embed_dim,
            use_conditioning_cross_attn=True,
            conditioning_cross_attn_layers=[2, 3, 4, 5],
            conditioning_num_heads=4,
            **unet_specific_params
        )
    
    def _build_image_conditioning_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build conditioning head for UNet."""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def get_shared_representation(self, control_rna, control_images, conditioning_info):
        rna_features = self.control_rna_encoder(control_rna)
        image_features = self.control_image_encoder(control_images)
        drug_embedding = self.drug_embedding(conditioning_info)

        # Drug participates in cross-modal attention so modality interaction
        # is conditioned on which perturbation is being modeled
        rna_enhanced, image_enhanced, attention_info = self.cross_modal_attention(
            rna_features, image_features, drug_features=drug_embedding
        )

        combined_features = torch.cat([rna_enhanced, image_enhanced, drug_embedding], dim=1)
        shared_repr = self.shared_encoder(combined_features)

        return {
            'shared_representation': shared_repr,
            'rna_features': rna_features,
            'image_features': image_features,
            'rna_enhanced': rna_enhanced,
            'image_enhanced': image_enhanced,
            'drug_embedding': drug_embedding,
            'attention_info': attention_info
        }

    def compute_cross_modal_consistency_loss(self, representations: Dict[str, torch.Tensor],
                                            treatment_rna: torch.Tensor = None,
                                            treatment_images: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Improved cross-modal consistency loss with stronger contrastive learning.
        Integrates alignment objectives directly with main task performance.
        """
        losses = {}
        
        rna_enhanced = representations['rna_enhanced']
        image_enhanced = representations['image_enhanced']
        shared_repr = representations['shared_representation']
        attention_info = representations['attention_info']
        
        # 1. STRONG CONTRASTIVE ALIGNMENT LOSS
        if 'contrastive_info' in attention_info:
            contrastive_info = attention_info['contrastive_info']
            losses['cross_modal_alignment'] = contrastive_info['contrastive_loss']
            losses['positive_similarity'] = contrastive_info['positive_similarity']
        else:
            # Fallback to simple alignment if contrastive info not available
            alignment_loss = self.compute_cosine_alignment_loss(rna_enhanced, image_enhanced)
            losses['cross_modal_alignment'] = alignment_loss

        # 3. ADAPTIVE VARIANCE REGULARIZATION
        variance_reg = self.compute_adaptive_variance_regularization(rna_enhanced, image_enhanced, losses.get('cross_modal_alignment', torch.tensor(0.0)))
        losses['variance_regularization'] = variance_reg
        
        return losses

    def compute_adaptive_variance_regularization(self, rna_enhanced: torch.Tensor,
                                            image_enhanced: torch.Tensor,
                                            alignment_loss: torch.Tensor,
                                            target_var: float = 1.0) -> torch.Tensor:
        """
        Adaptive variance regularization that only applies when it helps alignment.
        """
        # Compute current variances
        rna_var = torch.var(rna_enhanced, dim=0, unbiased=False).mean()
        image_var = torch.var(image_enhanced, dim=0, unbiased=False).mean()
        
        # Check for NaN and replace with target
        if torch.isnan(rna_var):
            rna_var = torch.tensor(target_var, device=rna_enhanced.device)
        if torch.isnan(image_var):
            image_var = torch.tensor(target_var, device=image_enhanced.device)
        
        # Only apply variance regularization if alignment is good
        # If alignment is poor, let variances adapt freely
        alignment_quality = torch.exp(-alignment_loss)
        adaptive_weight = alignment_quality * 0.01
        
        # Variance regularization  
        rna_var_loss = torch.abs(rna_var - target_var)
        image_var_loss = torch.abs(image_var - target_var)
        variance_reg = (rna_var_loss + image_var_loss) * adaptive_weight
        
        return variance_reg

    def compute_cosine_alignment_loss(self, rna_features: torch.Tensor, 
                                    image_features: torch.Tensor) -> torch.Tensor:
        """Fallback simple cosine similarity alignment loss."""
        # Normalize features
        rna_norm = F.normalize(rna_features, p=2, dim=1)
        image_norm = F.normalize(image_features, p=2, dim=1)
        
        # Compute element-wise cosine similarity
        cosine_sim = F.cosine_similarity(rna_norm, image_norm, dim=1)
        
        # Convert to loss (1 - similarity)
        alignment_loss = 1.0 - cosine_sim.mean()
        
        return alignment_loss
    
    def forward(self, x=None, t=None, control_rna=None, control_images=None,
                conditioning_info=None, mode='image'):

        if mode == 'train':
            # x is x_t (pre-noised by caller). Returns all outputs needed for loss
            # computation in a single DDP-coherent forward pass.
            representations = self.get_shared_representation(
                control_rna, control_images, conditioning_info
            )
            shared_repr = representations['shared_representation']
            pred_rna = self.transcriptome_direct_head(shared_repr)
            image_conditioning = self.image_conditioning_head(shared_repr)
            v_pred = self.unet(x, t, extra={"multimodal_conditioning": image_conditioning})
            return {
                'representations': representations,
                'pred_rna': pred_rna,
                'v_pred': v_pred
            }

        elif mode == 'transcriptome_direct':
            representations = self.get_shared_representation(
                control_rna, control_images, conditioning_info
            )
            return self.transcriptome_direct_head(representations['shared_representation'])

        elif mode == 'image':
            representations = self.get_shared_representation(
                control_rna, control_images, conditioning_info
            )
            shared_repr = representations['shared_representation']
            image_conditioning = self.image_conditioning_head(shared_repr)
            return self.unet(x, t, extra={"multimodal_conditioning": image_conditioning})

        elif mode == 'shared':
            return self.get_shared_representation(control_rna, control_images, conditioning_info)

        else:
            raise ValueError(f"Unknown mode: {mode}")

class RNAOnlyDrugConditionedModel(nn.Module):
    """
    RNA-only ablation: Control RNA + Drug → Treatment RNA only
    No image processing, no cross-modal attention.
    """
    def __init__(self,
                rna_dim: int,
                compound_vocab_size: int,
                cell_line_vocab_size: int,
                drug_embed_dim: int = 32,
                shared_embed_dim: int = 512,
                rna_output_dim: int = 256,
                gene_embed_dim: int = 64,
                rna_relation_rank: int = 25,
                rna_num_heads: int = 4,
                img_channels: int = 4,
                img_size: int = 256,
                **kwargs):
        super().__init__()
        
        self.rna_dim = rna_dim
        self.shared_embed_dim = shared_embed_dim
        # Store these for compatibility with checkpoint saving
        self.img_channels = img_channels
        self.img_size = img_size

        # Drug embedding
        self.drug_embedding = DrugEmbedding(
            compound_vocab_size=compound_vocab_size,
            cell_line_vocab_size=cell_line_vocab_size,
            embed_dim=drug_embed_dim
        )
        
        # RNA encoder only
        self.control_rna_encoder = RNAEncoder(
            input_dim=rna_dim,
            hidden_dims=[512, 256],
            output_dim=rna_output_dim,
            dropout=0.1,
            use_gene_relations=False,
            num_heads=rna_num_heads,
            relation_rank=rna_relation_rank,
            gene_embed_dim=gene_embed_dim
        )
        
        # Simplified shared encoder (RNA + Drug only)
        shared_input_dim = rna_output_dim + drug_embed_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_input_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            nn.LayerNorm(shared_embed_dim)
        )
        
        # RNA prediction head only
        self.transcriptome_direct_head = nn.Sequential(
            nn.Linear(shared_embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(1024, 512, dropout=0.1),
            nn.Linear(512, rna_dim)
        )
    
    def forward(self, x=None, t=None, control_rna=None, control_images=None, 
                conditioning_info=None, mode='transcriptome_direct'):
        """Forward pass for RNA-only model."""
        
        # Encode RNA and drug
        rna_features = self.control_rna_encoder(control_rna)
        drug_embedding = self.drug_embedding(conditioning_info)
        
        # Combine RNA + Drug into shared representation
        combined_features = torch.cat([rna_features, drug_embedding], dim=1)
        shared_repr = self.shared_encoder(combined_features)
        
        if mode == 'transcriptome_direct':
            return self.transcriptome_direct_head(shared_repr)
        elif mode == 'shared':
            return {'shared_representation': shared_repr}
        else:
            raise ValueError(f"RNA-only model doesn't support mode: {mode}")

class ImageOnlyDrugConditionedModel(nn.Module):
    """
    Image-only ablation: Control Images + Drug → Treatment Images only
    No RNA processing, no cross-modal attention.
    """
    def __init__(self,
                compound_vocab_size: int,
                cell_line_vocab_size: int,
                img_channels: int = 4,
                img_size: int = 256,
                model_channels: int = 128,
                drug_embed_dim: int = 32,
                shared_embed_dim: int = 512,
                image_output_dim: int = 256,
                rna_dim: int = None,
                # UNet parameters
                num_res_blocks: int = 2,
                attention_resolutions: list = None,
                dropout: float = 0.1,
                channel_mult: tuple = (1, 2, 2, 2),
                use_checkpoint: bool = False,
                num_heads: int = 4,
                num_head_channels: int = 32,
                use_scale_shift_norm: bool = True,
                resblock_updown: bool = True,
                use_new_attention_order: bool = True,
                **kwargs):  # Catch any extra params
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.shared_embed_dim = shared_embed_dim
        self.rna_dim = rna_dim or kwargs.get('rna_dim', 1000)

        # Set default for attention_resolutions if None
        if attention_resolutions is None:
            attention_resolutions = [16]

        # Drug embedding
        self.drug_embedding = DrugEmbedding(
            compound_vocab_size=compound_vocab_size,
            cell_line_vocab_size=cell_line_vocab_size,
            embed_dim=drug_embed_dim
        )
        
        # Image encoder only
        self.control_image_encoder = ImageEncoder(img_channels, image_output_dim)
        
        # Simplified shared encoder (Image + Drug only)
        shared_input_dim = image_output_dim + drug_embed_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_input_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            ResidualBlock(shared_embed_dim, shared_embed_dim, dropout=0.1),
            nn.LayerNorm(shared_embed_dim)
        )
        
        # Image conditioning and UNet only
        self.image_conditioning_head = nn.Sequential(
            nn.Linear(shared_embed_dim, shared_embed_dim),
            nn.SiLU(),
            nn.Linear(shared_embed_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim)
        )
        
        # UNet for image generation - only pass UNet-specific parameters
        self.unet = MultiModalConditionedUNet(
            in_channels=img_channels,
            model_channels=model_channels, 
            out_channels=img_channels,
            multimodal_embed_dim=shared_embed_dim,
            use_conditioning_cross_attn=True,
            conditioning_cross_attn_layers=[2, 3, 4, 5],
            conditioning_num_heads=4,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order
        )
    
    def forward(self, x=None, t=None, control_rna=None, control_images=None, 
                conditioning_info=None, mode='image'):
        """Forward pass for Image-only model."""
        
        # Encode image and drug
        image_features = self.control_image_encoder(control_images)
        drug_embedding = self.drug_embedding(conditioning_info)
        
        # Combine Image + Drug into shared representation
        combined_features = torch.cat([image_features, drug_embedding], dim=1)
        shared_repr = self.shared_encoder(combined_features)
        
        if mode == 'image':
            image_conditioning = self.image_conditioning_head(shared_repr)
            return self.unet(x, t, extra={"multimodal_conditioning": image_conditioning})
        elif mode == 'shared':
            return {'shared_representation': shared_repr}
        else:
            raise ValueError(f"Image-only model doesn't support mode: {mode}")