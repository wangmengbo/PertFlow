import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from encoder_kg import KnowledgeGraphDrugEncoder, KnowledgeGraphGeneEncoder, PrimeKGEncoder

class ResidualBlock(nn.Module):
    """Residual block with normalization and dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Skip connection with projection if dimensions don't match
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.main_branch(x) + self.skip(x)

class GeneMultiHeadAttention(nn.Module):
    """Multi-head self-attention for genes to attend to each other."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections for all heads at once
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, num_genes, embed_dim]
        """
        B, N, D = x.shape  # B=batch, N=num_genes, D=embed_dim
        
        # Generate Q, K, V for all heads
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        return self.out_proj(out), attn_weights

class RNAEncoder(nn.Module):
    """
    Encoder for RNA expression data with real self-attention over genes.
    Genes can dynamically attend to each other based on expression context.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=256, 
                dropout=0.1, use_gene_relations=False, num_heads=4, relation_rank=25,
                gene_embed_dim=512, num_attention_layers=1,
                use_kg=False, kg_processor=None, kg_data=None, 
                gene_to_kg_mapping=None, gene_names=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_gene_relations = use_gene_relations
        self.num_heads = num_heads
        self.relation_rank = relation_rank
        self.gene_embed_dim = gene_embed_dim
        self.use_kg = use_kg
        
        # ===== KG INTEGRATION =====
        if use_kg and kg_processor is not None and kg_data is not None and gene_to_kg_mapping is not None:
            # Create PrimeKGEncoder internally (same pattern as DrugEmbedding)
            kg_encoder = PrimeKGEncoder(
                node_features=kg_data['num_nodes_per_type'],
                relation_types=list(kg_data['edge_mappings'].keys()),
                hidden_dim=256,
                output_dim=128,
                num_layers=3
            )
            
            self.kg_gene_encoder = KnowledgeGraphGeneEncoder(
                kg_encoder=kg_encoder,
                gene_to_kg_mapping=gene_to_kg_mapping,
                gene_names=gene_names or [f"gene_{i}" for i in range(input_dim)],
                kg_embed_dim=128,
                output_dim=gene_embed_dim,
                kg_data=kg_data
            )
        else:
            self.kg_gene_encoder = None
        
        # Project raw gene expressions to embedding space for attention
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, gene_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(gene_embed_dim // 2, gene_embed_dim),
            nn.LayerNorm(gene_embed_dim)
        )

        self.gene_names = gene_names
        self.gene_relation_projection = nn.Linear(gene_embed_dim, 1)
        
        # Multi-layer self-attention for genes
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GeneMultiHeadAttention(gene_embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(gene_embed_dim),
                'norm2': nn.LayerNorm(gene_embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_attention_layers)
        ])

        # Low-rank gene relations
        if use_gene_relations:
            self.gene_relation_net_base = nn.Sequential(
                nn.Linear(gene_embed_dim * input_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.gene_relation_factors_head = nn.Linear(256, 2 * input_dim * self.relation_rank)

        # Pooling to get sample-level representation
        self.pooling_type = 'attention'
        if self.pooling_type == 'attention':
            self.pooling_attention = nn.Sequential(
                nn.Linear(gene_embed_dim, 1),
                nn.Tanh()
            )
        
        # Final encoder layers
        pooled_dim = gene_embed_dim
        layers = []
        prev_dim = pooled_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                ResidualBlock(prev_dim, hidden_dim, dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Final projection
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def apply_gene_relations(self, x_attended):
        """Apply learned gene-gene relationships using low-rank factorization."""
        batch_size, num_genes, embed_dim = x_attended.shape

        # Flatten attended features for relation learning
        x_flat = x_attended.view(batch_size, -1)  # [B, num_genes * embed_dim]
        
        # Get cell-specific embedding from attended gene features
        cell_embedding_for_relations = self.gene_relation_net_base(x_flat)

        # Predict parameters for U and V factor matrices
        relation_factors_params = self.gene_relation_factors_head(cell_embedding_for_relations)

        # Reshape to get U [B, G, K] and V [B, K, G] matrices per cell
        U = relation_factors_params[:, :num_genes * self.relation_rank].view(
            batch_size, num_genes, self.relation_rank
        )
        V = relation_factors_params[:, num_genes * self.relation_rank:].view(
            batch_size, self.relation_rank, num_genes
        )

        # Apply transformation to mean-pooled gene features for relations
        # gene_values = x_attended.mean(dim=-1)  # [B, G] - average over embedding dim
        gene_values = self.gene_relation_projection(x_attended).squeeze(-1)  # [B, G]
        gene_values_unsqueezed = gene_values.unsqueeze(1)  # [B, 1, G]
        temp = torch.bmm(gene_values_unsqueezed, U)  # [B, 1, K]
        gene_relations = torch.bmm(temp, V).squeeze(1)  # [B, G]
        
        # Apply relations back to attended features
        relation_weights = torch.sigmoid(gene_relations).unsqueeze(-1)  # [B, G, 1]
        return x_attended * (1 + 0.1 * relation_weights)

    def pool_gene_features(self, x):
        """Pool gene features to get sample-level representation."""
        if self.pooling_type == 'mean':
            return x.mean(dim=1)  # [B, embed_dim]
        elif self.pooling_type == 'max':
            return x.max(dim=1)[0]  # [B, embed_dim]
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_weights = self.pooling_attention(x)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            return (x * attn_weights).sum(dim=1)  # [B, embed_dim]

    def forward(self, x):
        """
        Forward pass with optional KG enhancement.
        x: [batch_size, num_genes] - raw gene expressions
        """
        batch_size, num_genes = x.shape
        
        # Embed each gene expression individually
        x_reshaped = x.unsqueeze(-1)  # [B, G, 1]
        gene_embeds = self.gene_embedding(x_reshaped)  # [B, G, embed_dim]
        
        # ===== KG ENHANCEMENT =====
        if self.use_kg and self.kg_gene_encoder is not None:
            if hasattr(self.kg_gene_encoder, 'get_kg_edge_dict'):
                edge_index_dict = self.kg_gene_encoder.get_kg_edge_dict()
            else:
                edge_index_dict = {}

            if hasattr(self, 'gene_names') and self.gene_names:
                assert len(self.gene_names) == num_genes, (
                    f"RNAEncoder has {len(self.gene_names)} gene names but received "
                    f"input with {num_genes} genes. Dataloader and model must use "
                    "identical ordered gene lists."
                )
                gene_subset = self.gene_names
            else:
                gene_subset = [f"gene_{i}" for i in range(num_genes)]

            kg_gene_embeds = self.kg_gene_encoder.get_gene_kg_embeddings(
                gene_subset=gene_subset,
                edge_index_dict=edge_index_dict
            )  # [num_genes, gene_embed_dim]

            # Broadcast to batch and add to expression embeddings
            gene_embeds = gene_embeds + 0.3 * kg_gene_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply multi-layer self-attention between genes
        attention_weights_all = []
        x_attended = gene_embeds
        
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out, attn_weights = layer['attention'](x_attended)
            x_attended = layer['norm1'](x_attended + attn_out)
            
            # Feed-forward with residual connection  
            ffn_out = layer['ffn'](x_attended)
            x_attended = layer['norm2'](x_attended + ffn_out)
            
            attention_weights_all.append(attn_weights)
        
        # Apply gene relations if enabled
        if self.use_gene_relations:
            x_attended = self.apply_gene_relations(x_attended)
        
        # Pool to get sample-level representation
        pooled_features = self.pool_gene_features(x_attended)  # [B, embed_dim]
        
        # Pass through encoder layers
        encoded_features = self.encoder(pooled_features)
        
        # Final projection
        final_embeddings = self.final_encoder(encoded_features)
        
        return final_embeddings
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            batch_size, num_genes = x.shape
            
            # Embed genes
            x_reshaped = x.unsqueeze(-1)
            gene_embeds = self.gene_embedding(x_reshaped)
            gene_embeds = gene_embeds + self.gene_position_embed.unsqueeze(0)
            
            # Get attention weights from each layer
            attention_weights_all = []
            x_attended = gene_embeds
            
            for layer in self.attention_layers:
                attn_out, attn_weights = layer['attention'](x_attended)
                x_attended = layer['norm1'](x_attended + attn_out)
                ffn_out = layer['ffn'](x_attended)
                x_attended = layer['norm2'](x_attended + ffn_out)
                attention_weights_all.append(attn_weights.cpu())
            
            return attention_weights_all  # List of [B, num_heads, N, N] tensors

class ResBlock(nn.Module):
    """ResNet-style residual block for image processing"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels or stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class ImageEncoder(nn.Module):
    """Much better replacement for your basic CNN encoder"""
    def __init__(self, img_channels=4, output_dim=256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual layers (like ResNet)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)      # [B, 64, H/4, W/4]
        x = self.layer1(x)    # [B, 64, H/4, W/4]
        x = self.layer2(x)    # [B, 128, H/8, W/8]
        x = self.layer3(x)    # [B, 256, H/16, W/16]
        x = self.layer4(x)    # [B, 512, H/32, W/32]
        
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.head(x)  # [B, output_dim]

class DrugEmbedding(nn.Module):
    """
    Drug embedding that combines experimental context with molecular features.
    Handles missing molecular data gracefully and maintains backward compatibility.
    Handles both tensor and list inputs from custom collate functions.
    """
    def __init__(self,
                compound_vocab_size: int,
                cell_line_vocab_size: int,
                embed_dim: int = 256,
                dropout: float = 0.1,
                max_smiles_length: int = 100,
                use_kg: bool = False,
                kg_processor=None,
                kg_data=None,
                drug_to_kg_mapping=None,
                compound_to_idx=None):   # add this
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_smiles_length = max_smiles_length
        self.use_kg = use_kg
        
        self.compound_embedding = nn.Embedding(compound_vocab_size, embed_dim)
        self.cell_line_embedding = nn.Embedding(cell_line_vocab_size, embed_dim)
        
        self.concentration_net = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.timepoint_net = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        fingerprint_dim = 2048
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        descriptor_2d_dim = 18
        descriptor_3d_dim = 5
        self.descriptor_encoder = nn.Sequential(
            nn.Linear(descriptor_2d_dim + descriptor_3d_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.smiles_char_embedding = nn.Embedding(128, 32)
        self.smiles_encoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.molecular_gate = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, 3),
            nn.Sigmoid()
        )
        
        if use_kg and kg_processor is not None and kg_data is not None:
            assert compound_to_idx is not None, \
                "compound_to_idx is required when use_kg=True"
            assert drug_to_kg_mapping is not None, \
                "drug_to_kg_mapping is required when use_kg=True"

            from encoder_kg import PrimeKGEncoder
            self.kg_encoder = PrimeKGEncoder(
                node_features=kg_data['num_nodes_per_type'],
                relation_types=list(kg_data['edge_mappings'].keys()),
                hidden_dim=256,
                output_dim=128,
                num_layers=3
            )
            
            self.drug_to_kg_mapping = drug_to_kg_mapping
            self.kg_data = kg_data

            for rel_type, data in kg_data['edge_data'].items():
                self.register_buffer(f'edge_index_{rel_type}', data['edge_index'])

            # Build compound vocab index → KG drug index mapping.
            # -1 means this compound is not in the KG.
            kg_idx_map = torch.full((compound_vocab_size,), -1, dtype=torch.long)
            for compound_name, vocab_idx in compound_to_idx.items():
                if compound_name in drug_to_kg_mapping:
                    kg_idx_map[vocab_idx] = drug_to_kg_mapping[compound_name]
            self.register_buffer('kg_idx_from_compound', kg_idx_map)

            kg_branch_dim = 128
        else:
            self.kg_encoder = None
            self.drug_to_kg_mapping = None
            self.kg_data = None
            kg_branch_dim = 0

        if self.use_kg:
            total_input_dim = 4 * embed_dim + 3 * embed_dim + 128
        else:
            total_input_dim = 4 * embed_dim + 3 * embed_dim

        self.fusion_net = nn.Sequential(
            nn.Linear(total_input_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _ensure_tensor(self, data, expected_shape, device):
        """Simplified tensor conversion."""
        if isinstance(data, list):
            if not data:
                return torch.zeros(expected_shape, device=device)
            
            # Try to stack directly first
            try:
                return torch.stack([torch.tensor(item, device=device) if not isinstance(item, torch.Tensor) 
                                else item.to(device) for item in data])
            except:
                # Fallback for variable shapes - just use zeros
                return torch.zeros(expected_shape, device=device)
        else:
            # Already a tensor
            return data.to(device)
    
    def _get_batch_size(self, drug_embeddings):
        """Get batch size from drug embeddings, handling both tensor and list cases."""
        for key, value in drug_embeddings.items():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    return len(value)
                else:
                    return len(value)
        raise ValueError("Cannot determine batch size from drug_embeddings")
    
    def encode_smiles(self, smiles_list):
        """
        Improved SMILES encoding with chemical tokenization.
        """
        batch_size = len(smiles_list)
        device = next(self.parameters()).device
        
        # Chemical vocabulary for better tokenization
        vocab = {
            '<PAD>': 0, '<UNK>': 1,
            'C': 2, 'N': 3, 'O': 4, 'S': 5, 'P': 6, 'F': 7,
            'Cl': 8, 'Br': 9, 'I': 10,
            '(': 11, ')': 12, '[': 13, ']': 14,
            '=': 15, '#': 16, '-': 17, '+': 18, '@': 19,
            '0': 20, '1': 21, '2': 22, '3': 23, '4': 24,
            '5': 25, '6': 26, '7': 27, '8': 28, '9': 29
        }
        
        def tokenize_smiles(smiles):
            """Better SMILES tokenization."""
            tokens = []
            i = 0
            while i < len(smiles):
                # Check two-character tokens first
                if i < len(smiles) - 1:
                    two_char = smiles[i:i+2]
                    if two_char in vocab:
                        tokens.append(vocab[two_char])
                        i += 2
                        continue
                
                # Single character
                char = smiles[i]
                tokens.append(vocab.get(char, vocab['<UNK>']))
                i += 1
            return tokens
        
        # Tokenize all SMILES
        tokenized = []
        for smiles in smiles_list:
            if smiles and smiles.strip():
                tokens = tokenize_smiles(smiles)[:self.max_smiles_length]
            else:
                tokens = [vocab['<PAD>']]
            tokenized.append(tokens)
        
        # Pad sequences
        max_len = min(max(len(tokens) for tokens in tokenized) if tokenized else 1, self.max_smiles_length)
        smiles_tensor = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, tokens in enumerate(tokenized):
            length = min(len(tokens), max_len)
            smiles_tensor[i, :length] = torch.tensor(tokens[:length], device=device)
        
        # Embed and encode (keep your existing embedding + conv approach)
        char_embeds = self.smiles_char_embedding(smiles_tensor)
        char_embeds = char_embeds.transpose(1, 2)
        smiles_encoded = self.smiles_encoder(char_embeds)
        
        return smiles_encoded
    
    def process_molecular_features(self, drug_embeddings):
        """
        Process molecular features with availability-based gating.
        Handles both tensor and list inputs from custom collate functions.
        
        Args:
            drug_embeddings: Dictionary containing batched molecular features from dataloader
            
        Returns:
            Gated molecular features [batch_size, 3 * embed_dim]
        """
        device = next(self.parameters()).device
        
        # Get batch size safely
        batch_size = self._get_batch_size(drug_embeddings)
        
        # Extract availability flags - handle both tensor and list cases
        has_fingerprints = torch.ones(batch_size, dtype=torch.bool, device=device)
        if 'fingerprint_morgan' in drug_embeddings:
            morgan_data = drug_embeddings['fingerprint_morgan']
            rdkit_data = drug_embeddings['fingerprint_rdkit']
            
            # Convert to tensors if needed
            morgan_tensor = self._ensure_tensor(morgan_data, (batch_size, 1024), device)
            rdkit_tensor = self._ensure_tensor(rdkit_data, (batch_size, 1024), device)
            
            # Check if fingerprints are non-zero
            morgan_nonzero = (morgan_tensor.sum(dim=1) != 0)
            rdkit_nonzero = (rdkit_tensor.sum(dim=1) != 0)
            has_fingerprints = (morgan_nonzero | rdkit_nonzero)

        has_fingerprints = has_fingerprints.float()
        
        # Handle boolean flags properly
        has_2d_data = drug_embeddings.get('has_2d_structure', [True] * batch_size)
        if isinstance(has_2d_data, torch.Tensor):
            has_2d = has_2d_data.clone().detach().bool().to(device)
        elif isinstance(has_2d_data, list):
            has_2d = torch.tensor(has_2d_data, dtype=torch.bool, device=device)
        else:
            has_2d = torch.tensor([has_2d_data] * batch_size, dtype=torch.bool, device=device)

        has_3d_data = drug_embeddings.get('has_3d_structure', [True] * batch_size)
        if isinstance(has_3d_data, torch.Tensor):
            has_3d = has_3d_data.clone().detach().bool().to(device)
        elif isinstance(has_3d_data, list):
            has_3d = torch.tensor(has_3d_data, dtype=torch.bool, device=device)
        else:
            has_3d = torch.tensor([has_3d_data] * batch_size, dtype=torch.bool, device=device)

        has_descriptors = (has_2d | has_3d).float()
        
        has_smiles = torch.ones(batch_size, dtype=torch.bool, device=device)
        if 'smiles' in drug_embeddings:
            smiles_data = drug_embeddings['smiles']
            if isinstance(smiles_data, list):
                has_smiles = torch.tensor([
                    bool(smiles and smiles.strip() != '') 
                    for smiles in smiles_data
                ], dtype=torch.bool, device=device)

        has_smiles = has_smiles.float()
        
        # Stack availability flags
        availability_flags = torch.stack([has_fingerprints, has_descriptors, has_smiles], dim=1)
        
        # Compute gating weights
        gate_weights = self.molecular_gate(availability_flags)  # [B, 3]
        
        # Process fingerprints (Morgan + RDKit combined)
        if 'fingerprint_morgan' in drug_embeddings and 'fingerprint_rdkit' in drug_embeddings:
            morgan_fp = self._ensure_tensor(
                drug_embeddings['fingerprint_morgan'], 
                (batch_size, 1024), 
                device
            )
            rdkit_fp = self._ensure_tensor(
                drug_embeddings['fingerprint_rdkit'], 
                (batch_size, 1024), 
                device
            )
            combined_fp = torch.cat([morgan_fp, rdkit_fp], dim=1)
        else:
            combined_fp = torch.zeros(batch_size, 2048, device=device)
        
        fingerprint_features = self.fingerprint_encoder(combined_fp)
        
        # Process descriptors (2D + 3D combined)
        if 'descriptors_2d' in drug_embeddings:
            desc_2d = self._ensure_tensor(
                drug_embeddings['descriptors_2d'], 
                (batch_size, 18), 
                device
            )
            desc_3d_data = drug_embeddings.get('descriptors_3d', torch.zeros(batch_size, 5))
            desc_3d = self._ensure_tensor(desc_3d_data, (batch_size, 5), device)
            combined_desc = torch.cat([desc_2d, desc_3d], dim=1)
        else:
            combined_desc = torch.zeros(batch_size, 23, device=device)
        
        descriptor_features = self.descriptor_encoder(combined_desc)
        
        # Process SMILES
        smiles_list = drug_embeddings.get('smiles', [''] * batch_size)
        smiles_features = self.encode_smiles(smiles_list)
        
        # Apply gating
        gated_fingerprint = fingerprint_features * gate_weights[:, 0:1]
        gated_descriptor = descriptor_features * gate_weights[:, 1:2]
        gated_smiles = smiles_features * gate_weights[:, 2:3]
        
        # Concatenate molecular features
        molecular_features = torch.cat([
            gated_fingerprint, 
            gated_descriptor, 
            gated_smiles
        ], dim=1)
        
        return molecular_features

    def get_kg_embeddings(self, conditioning_info: Dict) -> torch.Tensor:
        """
        Get KG drug embeddings for a batch using compound_ids directly.
        Single full-graph GNN forward, then index by pre-built vocab→KG mapping.
        """
        device = next(self.parameters()).device
        compound_ids = conditioning_info['compound_ids']  # [batch_size], always present
        batch_size = compound_ids.shape[0]

        if not self.use_kg or self.kg_encoder is None:
            return torch.zeros(batch_size, 128, device=device)

        edge_dict = {
            rel_type: getattr(self, f'edge_index_{rel_type}')
            for rel_type in self.kg_data['edge_data'].keys()
        }

        # Single full-graph forward — all node types, all edges
        all_embeddings = self.kg_encoder(edge_dict)
        drug_all = all_embeddings.get('drug')  # [num_kg_drugs, 128]

        if drug_all is None:
            return torch.zeros(batch_size, 128, device=device)

        # Map compound vocab indices to KG indices; -1 means not in KG
        kg_indices = self.kg_idx_from_compound[compound_ids]  # [batch_size]
        in_kg = kg_indices >= 0  # [batch_size]

        # Clamp so we can index safely; zero out non-KG entries after
        safe_indices = kg_indices.clamp(min=0)
        embs = drug_all[safe_indices]  # [batch_size, 128]
        embs = embs * in_kg.unsqueeze(1).float()

        return embs
            
    def forward(self, conditioning_info: Dict, drug_embeddings: Optional[Dict] = None) -> torch.Tensor:
        # ===== EXPERIMENTAL CONTEXT BRANCH =====
        compound_emb = self.compound_embedding(conditioning_info['compound_ids'])
        cell_line_emb = self.cell_line_embedding(conditioning_info['cell_line_ids'])
        
        concentration_emb = self.concentration_net(conditioning_info['concentration'].unsqueeze(-1))
        timepoint_emb = self.timepoint_net(conditioning_info['timepoint'].unsqueeze(-1))
        
        experimental_features = torch.cat([
            compound_emb, 
            cell_line_emb, 
            concentration_emb, 
            timepoint_emb
        ], dim=-1)
        
        # ===== MOLECULAR STRUCTURE BRANCH =====
        molecular_data = None
        if drug_embeddings is not None:
            molecular_data = drug_embeddings
        elif 'drug_embeddings' in conditioning_info:
            molecular_data = conditioning_info['drug_embeddings']
        
        if molecular_data is not None:
            molecular_features = self.process_molecular_features(molecular_data)
        else:
            batch_size = compound_emb.shape[0]
            device = compound_emb.device
            molecular_features = torch.zeros(batch_size, 3 * self.embed_dim, device=device)
        
        # ===== KG INTEGRATION =====
        if self.use_kg and self.kg_encoder is not None:
            kg_features = self.get_kg_embeddings(conditioning_info)
        elif self.use_kg:
            kg_features = torch.zeros(compound_emb.shape[0], 128, device=compound_emb.device)
        else:
            kg_features = None

        # ===== FUSION =====
        if kg_features is not None:
            combined_features = torch.cat([experimental_features, molecular_features, kg_features], dim=1)
        else:
            combined_features = torch.cat([experimental_features, molecular_features], dim=1)

        return self.fusion_net(combined_features)