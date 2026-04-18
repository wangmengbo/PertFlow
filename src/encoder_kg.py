# kg_integration.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import logging
from dataloader import PrimeKGProcessor

logger = logging.getLogger(__name__)


class HeterogeneousRelationConv(nn.Module):
    """
    Proper heterogeneous GNN layer using PyG HeteroConv.
    Handles cross-type edges (drug→protein, drug→effect) via SAGEConv bipartite message passing.
    """

    EDGE_TRIPLES = {
        'drug_protein':     ('drug', 'drug_protein',     'gene/protein'),
        'drug_drug':        ('drug', 'drug_drug',         'drug'),
        'protein_protein':  ('gene/protein', 'protein_protein', 'gene/protein'),
        'drug_effect':      ('drug', 'drug_effect',       'effect/phenotype'),
    }

    def __init__(self, hidden_dim: int, relation_types: List[str], conv_type: str = 'sage'):
        super().__init__()

        conv_dict = {}
        for rel_type in relation_types:
            if rel_type not in self.EDGE_TRIPLES:
                logger.warning(f"Unknown relation type: {rel_type}, skipping")
                continue
            triple = self.EDGE_TRIPLES[rel_type]
            if conv_type in ('gcn', 'sage'):
                # SAGEConv handles bipartite (src_type != dst_type) natively with tuple in_channels
                conv_dict[triple] = SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr='mean')
            elif conv_type == 'gat':
                conv_dict[triple] = GATConv(
                    (hidden_dim, hidden_dim), hidden_dim,
                    heads=4, concat=False, add_self_loops=False
                )

        self.hetero_conv = HeteroConv(conv_dict, aggr='sum')
        self.active_triples = set(conv_dict.keys())

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        triple_edge_dict = {}
        for rel_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            if rel_type not in self.EDGE_TRIPLES:
                continue
            triple = self.EDGE_TRIPLES[rel_type]
            if triple in self.active_triples:
                triple_edge_dict[triple] = edge_index

        if not triple_edge_dict:
            return {}

        # Only pass node types that appear in at least one active edge
        active_node_types = set()
        for src, _, dst in triple_edge_dict:
            active_node_types.add(src)
            active_node_types.add(dst)

        active_x = {k: v for k, v in x_dict.items() if k in active_node_types and v.shape[0] > 0}

        return self.hetero_conv(active_x, triple_edge_dict)


class PrimeKGEncoder(nn.Module):
    """
    Heterogeneous GNN encoder for PrimeKG.
    Extracts drug and gene/protein embeddings from the knowledge graph.
    """
    
    def __init__(self, 
                 node_features: Dict[str, int],
                 relation_types: List[str],
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 3,
                 conv_type: str = 'gcn',
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_types = list(node_features.keys())
        self.relation_types = relation_types
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Initial node embeddings (learnable)
        self.node_embeddings = nn.ModuleDict()
        for node_type, num_nodes in node_features.items():
            self.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        
        self.conv_layers = nn.ModuleList([
            HeterogeneousRelationConv(
                hidden_dim=hidden_dim,
                relation_types=relation_types,
                conv_type=conv_type
            )
            for _ in range(num_layers)
        ])
        
        # Output projection layers
        self.output_projections = nn.ModuleDict()
        for node_type in self.node_types:
            self.output_projections[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full-graph forward pass. All nodes of every type participate in message passing.
        Returns output embeddings indexed by node type, covering all nodes.

        Args:
            edge_index_dict: string-keyed edge indices, e.g. {'drug_protein': tensor([2, N])}

        Returns:
            Dict mapping node_type -> [num_nodes_of_type, output_dim]
        """
        device = next(self.parameters()).device

        # Embed ALL nodes for each type — necessary for cross-type message passing
        x_dict: Dict[str, torch.Tensor] = {}
        for node_type, emb_layer in self.node_embeddings.items():
            indices = torch.arange(emb_layer.num_embeddings, device=device)
            x_dict[node_type] = emb_layer(indices)  # [num_nodes, hidden_dim]

        # Graph convolutions with residual connections
        for conv_layer in self.conv_layers:
            x_new = conv_layer(x_dict, edge_index_dict)
            for node_type in x_dict:
                if node_type in x_new:
                    x_dict[node_type] = self.dropout(F.relu(x_dict[node_type] + x_new[node_type]))

        # Output projections
        output: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            if node_type in self.output_projections:
                output[node_type] = self.output_projections[node_type](x)

        return output
    
    def get_drug_embeddings(self, drug_indices: torch.Tensor, 
                           edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embeddings specifically for drugs."""
        node_indices = {'drug': drug_indices}
        
        # Add dummy indices for other node types if needed
        for node_type in self.node_types:
            if node_type != 'drug' and node_type not in node_indices:
                node_indices[node_type] = torch.tensor([], dtype=torch.long, device=drug_indices.device)
        
        embeddings = self.forward(node_indices, edge_index_dict)
        return embeddings.get('drug', torch.zeros((len(drug_indices), self.output_dim)))
    
    def get_gene_embeddings(self, gene_indices: torch.Tensor,
                        edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embeddings specifically for genes/proteins."""
        if len(gene_indices) == 0:
            return torch.zeros((0, self.output_dim), device=gene_indices.device)
        
        # Only include the specific gene indices we need
        node_indices = {'gene/protein': gene_indices}
        
        # Add empty tensors for other node types
        device = gene_indices.device
        for node_type in self.node_types:
            if node_type != 'gene/protein':
                node_indices[node_type] = torch.tensor([], dtype=torch.long, device=device)
        
        embeddings = self.forward(node_indices, edge_index_dict)
        return embeddings.get('gene/protein', torch.zeros((len(gene_indices), self.output_dim), device=device))


class KnowledgeGraphDrugEncoder(nn.Module):
    """
    Enhanced drug encoder that combines existing molecular features 
    with knowledge graph embeddings.
    """
    
    def __init__(self, 
                 original_drug_embedding: nn.Module,
                 kg_processor: PrimeKGProcessor,
                 kg_data: Dict,
                 drug_to_kg_mapping: Dict[str, int],
                 kg_embed_dim: int = 128,
                 fusion_dim: int = 512):
        super().__init__()
        
        self.original_drug_embedding = original_drug_embedding
        self.drug_to_kg_mapping = drug_to_kg_mapping
        self.kg_embed_dim = kg_embed_dim
        
        # Initialize KG encoder
        self.kg_encoder = PrimeKGEncoder(
            node_features=kg_data['num_nodes_per_type'],
            relation_types=list(kg_data['edge_mappings'].keys()),
            hidden_dim=256,
            output_dim=kg_embed_dim,
            num_layers=3
        )
        
        # Store edge indices as buffers (non-trainable)
        self.edge_data = kg_data['edge_data']
        for rel_type, data in self.edge_data.items():
            self.register_buffer(f'edge_index_{rel_type}', data['edge_index'])
        
        # Fusion network to combine original + KG embeddings
        original_dim = original_drug_embedding.embed_dim
        combined_dim = original_dim + kg_embed_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, original_dim),  # Keep same output dim as original
            nn.LayerNorm(original_dim)
        )
        
        # Learnable weight for balancing original vs KG features
        self.kg_weight = nn.Parameter(torch.tensor(0.3))
        
    def get_kg_edge_dict(self) -> Dict[str, torch.Tensor]:
        """Get edge indices for all relation types."""
        edge_dict = {}
        for rel_type in self.edge_data.keys():
            edge_dict[rel_type] = getattr(self, f'edge_index_{rel_type}')
        return edge_dict
    
    def forward(self, conditioning_info: Dict, drug_embeddings: Optional[Dict] = None) -> torch.Tensor:
        """
        Enhanced forward pass that combines original molecular features with KG embeddings.
        """
        device = next(self.parameters()).device
        
        # Get original drug embeddings
        original_embeddings = self.original_drug_embedding(conditioning_info, drug_embeddings)
        batch_size = original_embeddings.shape[0]
        
        # Get compound names from conditioning info
        if 'compound_name' in conditioning_info:
            compound_names = conditioning_info['compound_name']
        elif 'treatment' in conditioning_info:
            compound_names = conditioning_info['treatment']
        else:
            # Fallback: just use original embeddings
            return original_embeddings
        
        # Get KG embeddings for drugs in this batch
        kg_embeddings_list = []
        
        for compound_name in compound_names:
            if compound_name in self.drug_to_kg_mapping:
                kg_idx = self.drug_to_kg_mapping[compound_name]
                drug_indices = torch.tensor([kg_idx], device=device)
                
                # Get KG embedding for this drug
                kg_emb = self.kg_encoder.get_drug_embeddings(
                    drug_indices, self.get_kg_edge_dict()
                )
                kg_embeddings_list.append(kg_emb.squeeze(0))
            else:
                # No KG embedding available, use zeros
                kg_embeddings_list.append(torch.zeros(self.kg_embed_dim, device=device))
        
        # Stack KG embeddings
        kg_embeddings = torch.stack(kg_embeddings_list)  # [batch_size, kg_embed_dim]
        
        # Combine original and KG embeddings
        combined_embeddings = torch.cat([original_embeddings, kg_embeddings], dim=1)
        
        # Apply fusion network
        fused_embeddings = self.fusion_network(combined_embeddings)
        
        # Weighted combination with original embeddings
        alpha = torch.sigmoid(self.kg_weight)
        final_embeddings = (1 - alpha) * original_embeddings + alpha * fused_embeddings
        
        return final_embeddings


class KnowledgeGraphGeneEncoder(nn.Module):
    """
    Gene encoder that uses knowledge graph embeddings for genes/proteins.
    Can be integrated with your RNA encoder.
    """
    
    def __init__(self,
                 kg_encoder: PrimeKGEncoder,
                 gene_to_kg_mapping: Dict[str, int],
                 gene_names: List[str],
                 kg_embed_dim: int = 128,
                 output_dim: int = 256,
                 kg_data: Optional[Dict] = None):  # Add kg_data parameter
        super().__init__()
        
        self.kg_encoder = kg_encoder
        self.gene_to_kg_mapping = gene_to_kg_mapping
        self.gene_names = gene_names
        self.kg_embed_dim = kg_embed_dim
        self.output_dim = output_dim
        
        # Store edge data if provided
        if kg_data is not None:
            self.edge_data = kg_data['edge_data']
            # Store edge indices as buffers
            for rel_type, data in self.edge_data.items():
                self.register_buffer(f'edge_index_{rel_type}', data['edge_index'])
        else:
            self.edge_data = {}
        
        # Create gene index mapping
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        # Projection layer for KG gene embeddings
        self.gene_projection = nn.Sequential(
            nn.Linear(kg_embed_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Fallback embeddings for genes not in KG
        self.fallback_embeddings = nn.Embedding(len(gene_names), output_dim)
    
    def get_kg_edge_dict(self) -> Dict[str, torch.Tensor]:
        """Get edge indices for all relation types."""
        edge_dict = {}
        for rel_type in self.edge_data.keys():
            if hasattr(self, f'edge_index_{rel_type}'):
                edge_dict[rel_type] = getattr(self, f'edge_index_{rel_type}')
        return edge_dict
        
    def get_gene_kg_embeddings(self, gene_subset: List[str],
                               edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get projected KG embeddings for a list of gene names.
        Single full-graph GNN forward pass; index into results by gene name.
        Falls back to learned embeddings for genes absent from KG.
        """
        device = next(self.parameters()).device
        n = len(gene_subset)

        # Single full-graph forward — all node types participate in message passing
        all_embeddings = self.kg_encoder(edge_index_dict)
        gene_all = all_embeddings.get('gene/protein')  # [num_kg_genes, kg_embed_dim] or None

        in_kg = torch.tensor(
            [g in self.gene_to_kg_mapping for g in gene_subset],
            dtype=torch.bool, device=device
        )  # [n]

        # Use index 0 as placeholder for genes not in KG (masked out below)
        kg_indices = torch.tensor(
            [self.gene_to_kg_mapping.get(g, 0) for g in gene_subset],
            device=device
        )  # [n]

        if gene_all is not None:
            kg_embs = self.gene_projection(gene_all[kg_indices])  # [n, output_dim]
        else:
            kg_embs = torch.zeros(n, self.output_dim, device=device)

        # Fallback: learned per-gene embedding for genes not in KG
        local_indices = torch.tensor(
            [self.gene_to_idx.get(g, 0) for g in gene_subset],
            device=device
        )
        fallback_embs = self.fallback_embeddings(local_indices)  # [n, output_dim]

        return torch.where(in_kg.unsqueeze(1), kg_embs, fallback_embs)
