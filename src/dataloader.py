import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tifffile
import pickle
import h5py
import json
import logging
from typing import Dict, List, Optional, Callable, Tuple
from torchvision import transforms
import scanpy as sc
import anndata as ad
import networkx as nx
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
from scipy.ndimage import zoom

import warnings
os.environ['RDKit_SILENCE_WARNINGS'] = '1'
import rdkit
rdkit.rdBase.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drug_process import MultiModalDrugDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_array(data, name, allow_negative=True, max_abs_value=1e6):
    """Comprehensive data validation helper."""
    if data is None:
        raise ValueError(f"{name}: Data is None")
    
    # Check for NaNs
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise ValueError(f"{name}: Contains {nan_count} NaN values")
    
    # Check for infinite values
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        raise ValueError(f"{name}: Contains {inf_count} infinite values")
    
    # Check for all zeros (might indicate loading issues)
    if np.all(data == 0):
        raise ValueError(f"{name}: All values are zero")
    
    return True


def scale_down_images(images: np.ndarray, target_size: int) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    c, h, w = images.shape
    return zoom(images, (1, target_size / h, target_size / w), order=1)


class HistologyTranscriptomicsDataset(Dataset):
    """
    Custom Dataset for paired histology images and transcriptomics data
    with drug treatment conditioning.
    """
    
    def __init__(self, 
                 metadata_control: pd.DataFrame,
                 metadata_drug: pd.DataFrame, 
                 gene_count_matrix: pd.DataFrame,
                 image_json_dict: Dict[str, List[str]],
                 transform: Optional[Callable] = None,
                 target_size: Optional[int] = None,
                 ):
        """
        Args:
            metadata_control: Control dataset metadata with columns ['cell_line', 'sample_id', 'json_key']
            metadata_drug: Treatment dataset metadata with columns ['cell_line', 'compound', 'timepoint', 
                          'compound_concentration_in_uM', 'sample_id', 'json_key']
            gene_count_matrix: Transcriptomics data with sample_id as columns and genes as rows
            image_json_dict: Dictionary mapping json_key to list of image paths
            transform: Optional transform to be applied on images
            target_size: Optional target size for image resizing
        """
        self.metadata_control = metadata_control
        self.metadata_drug = metadata_drug
        self.gene_count_matrix = gene_count_matrix
        self.image_json_dict = image_json_dict
        self.transform = transform
        self.target_size = target_size

        # Convert relevant columns to appropriate types
        for k in ['sample_id', 'cell_line', 'json_key', 'compound']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(str)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(str)
        
        for k in ['timepoint', 'compound_concentration_in_uM']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(float)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(float)

        # Group control metadata by cell_line for efficient sampling
        self.control_grouped = self.metadata_control.groupby('cell_line')
        
        # Get available cell lines in both datasets
        control_cell_lines = set(self.metadata_control['cell_line'].unique())
        drug_cell_lines = set(self.metadata_drug['cell_line'].unique())
        self.common_cell_lines = control_cell_lines.intersection(drug_cell_lines)
        
        # Filter drug metadata to only include samples with matching control cell lines
        self.filtered_drug_metadata = self.metadata_drug[
            self.metadata_drug['cell_line'].isin(self.common_cell_lines)
        ].reset_index(drop=True)
        
        logger.info(f"Dataset initialized with {len(self.filtered_drug_metadata)} treatment samples")
        logger.info(f"Common cell lines: {len(self.common_cell_lines)}")
    
    def __len__(self):
        return len(self.filtered_drug_metadata)
    
    def load_multi_channel_images(self, json_key: str) -> np.ndarray:
        """
        Load all TIFF images for a sample and concatenate as 3D array.
        
        Args:
            json_key: Key to locate image paths in the JSON dictionary
            
        Returns:
            3D numpy array of shape (channels, height, width)
        """
        image_paths = self.image_json_dict.get(json_key, [])
        if not image_paths:
            raise ValueError(f"No images found for json_key: {json_key}")
        
        # Sort paths to ensure consistent channel order (w1, w2, w3, w4)
        image_paths = sorted(image_paths)
        
        images = []
        for i, path in enumerate(image_paths):
            try:
                img = tifffile.imread(path)
                # Ensure 2D image (H, W)
                if img.ndim > 2:
                    img = img.squeeze()

                images.append(img)
                
                # Log channel information for debugging
                channel_info = "w1=Blue" if "w1" in path else \
                              "w2=Green" if "w2" in path else \
                              "w3=Red" if "w3" in path else \
                              "w4=DeepRed" if "w4" in path else "Unknown"
                logger.debug(f"Loaded channel {i}: {channel_info} from {path}")
                
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                raise
        
        # Stack images along the channel dimension to form (C, H, W)
        images = np.stack(images, axis=0).astype(np.float32)
        
        # Scale images if target_size is provided
        if self.target_size is not None:
            images = scale_down_images(images, self.target_size)
        
        # Apply transform if provided
        if self.transform:
            images = self.transform(images)
        
        return images
    
    def get_transcriptomics_data(self, sample_id: str, normalize: bool = False) -> np.ndarray:
        """
        Extract transcriptomics data for a given sample_id.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            1D numpy array of gene expression values
        """
        if sample_id not in self.gene_count_matrix.columns:
            logger.error(f"self.gene_count_matrix.columns[:10]={self.gene_count_matrix.columns[:10].tolist()}")
            logger.error(f"\"{sample_id}\" in self.gene_count_matrix.columns={sample_id in self.gene_count_matrix.columns}")
            logger.error(f"gene_count_matrix.shape={self.gene_count_matrix.shape}")
            raise ValueError(f"Sample ID {sample_id} not found in gene count matrix")
        
        if not normalize:
            return self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        raw_data = self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        # Apply the SAME transformation as the encoder
        log_data = np.log1p(raw_data)  # Log transform

        # Normalize (store global stats for consistency)
        if not hasattr(self, 'global_mean'):
            # Compute global statistics once
            all_log_data = np.log1p(self.gene_count_matrix.values)
            self.global_mean = np.mean(all_log_data)
            self.global_std = np.std(all_log_data)
        
        normalized_data = (log_data - self.global_mean) / (self.global_std + 1e-8)
        
        return normalized_data.astype(np.float32)  # Ensure float32 for consistency
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired sample (control + treatment) with conditioning information.
        
        Args:
            idx: Index of the treatment sample
            
        Returns:
            Dictionary containing paired data and conditioning information
        """
        # Get treatment sample metadata
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        cell_line = treatment_sample['cell_line']
        
        # logger.debug(f"treatment_sample['sample_id']=\"{treatment_sample['sample_id']}\"")
        logger.debug(f"treatment_sample={treatment_sample.to_dict()}")
        
        # Sample a random control sample with the same cell_line
        control_samples = self.control_grouped.get_group(cell_line)
        control_sample = control_samples.sample(n=1).iloc[0]
        # logger.debug(f"control_sample=\"{control_sample['sample_id']}\"")
        logger.debug(f"control_sample={control_sample.to_dict()}")
        
        # Load transcriptomics data
        treatment_transcriptomics = self.get_transcriptomics_data(treatment_sample['sample_id'])
        control_transcriptomics = self.get_transcriptomics_data(control_sample['sample_id'])
        
        # Load multi-channel images
        treatment_images = self.load_multi_channel_images(treatment_sample['json_key'])
        control_images = self.load_multi_channel_images(control_sample['json_key'])
        
        # Prepare conditioning information
        conditioning_info = {
            'treatment': treatment_sample['compound'],
            'cell_line': cell_line,
            'timepoint': treatment_sample['timepoint'],
            'compound_concentration_in_uM': treatment_sample['compound_concentration_in_uM']
        }

        if np.isnan(conditioning_info['timepoint']) or np.isnan(conditioning_info['compound_concentration_in_uM']):
            raise ValueError(f"NaN in conditioning info: {conditioning_info}")

        # Return paired data as tensors (CORRECTED - fixed image assignment)
        return {
            'control_transcriptomics': torch.tensor(control_transcriptomics),
            'treatment_transcriptomics': torch.tensor(treatment_transcriptomics),
            'control_images': torch.tensor(control_images),      # FIXED
            'treatment_images': torch.tensor(treatment_images),  # FIXED
            'conditioning_info': conditioning_info
        }


def create_vocab_mappings(dataset):
    """Create vocabulary mappings for categorical variables."""
    compounds = list(set(dataset.metadata_control['compound'].unique()).union(set(dataset.metadata_drug['compound'].unique())))
    cell_lines = list(set(dataset.metadata_control['cell_line'].unique()).union(set(dataset.metadata_drug['cell_line'].unique())))
    
    compound_to_idx = {comp: idx for idx, comp in enumerate(sorted(compounds))}
    cell_line_to_idx = {cl: idx for idx, cl in enumerate(sorted(cell_lines))}
    
    return compound_to_idx, cell_line_to_idx


def image_transform(images):
    """
    Transform for 16-bit multi-channel microscopy images.
    Args:
        images: numpy array of shape (channels, height, width)
    
    Returns:
        Normalized and contrast-enhanced images
    """
    # Normalize 16-bit to 0-1 range (CORRECTED from /255 to /65535)
    images_norm = (images / 32767.5) - 1.0
    # Apply per-channel contrast enhancement
    enhanced_images = np.zeros_like(images_norm)
    for i in range(images_norm.shape[0]):
        channel = images_norm[i]
        p1, p99 = np.percentile(channel, [1, 99])
        if p99 > p1:
            enhanced_images[i] = np.clip((channel - p1) / (p99 - p1) * 2 - 1, -1, 1)
        else:
            enhanced_images[i] = channel
    return enhanced_images.astype(np.float32)


class DatasetWithDrugs(HistologyTranscriptomicsDataset):
    """
    Dataset that includes drug conditioning information.
    """
    
    def __init__(self,
                 metadata_control: pd.DataFrame,
                 metadata_drug: pd.DataFrame,
                 gene_count_matrix: pd.DataFrame,
                 image_json_dict: Dict[str, List[str]],
                 drug_data_path: str,

                 transform: Optional[Callable] = None,
                 target_size: Optional[int] = None,
                 drug_encoder: Optional[torch.nn.Module] = None,
                 debug_mode: bool = False,
                 debug_samples: int = 50,
                 debug_cell_lines: Optional[List[str]] = None,
                 debug_drugs: Optional[List[str]] = None,
                 exclude_drugs: Optional[List[str]] = None,

                 fallback_smiles_dict=None,
                 enable_smiles_fallback=True,
                 ):
        """
        Args:
            drug_data_path: Path to preprocessed drug embeddings file
            drug_encoder: Optional drug encoder model (if None, uses raw embeddings)
            debug_mode: If True, only load a subset of data for debugging
            debug_samples: Number of samples to load in debug mode
            debug_cell_lines: Specific cell lines to use for debugging (optional)
        """
        # Store debug parameters
        self.debug_mode = debug_mode
        self.debug_samples = debug_samples
        self.debug_cell_lines = debug_cell_lines
        
        # Apply debug filtering to metadata BEFORE parent initialization
        if debug_mode:
            original_drug_size = len(metadata_drug)
            original_control_size = len(metadata_control)
            
            # Filter by specific cell lines if provided
            if debug_cell_lines:
                metadata_drug = metadata_drug[metadata_drug['cell_line'].isin(debug_cell_lines)]
                metadata_control = metadata_control[metadata_control['cell_line'].isin(debug_cell_lines)]
                print(f"DEBUG MODE: Filtered to cell lines: {debug_cell_lines}")
            
            if debug_drugs:
                metadata_drug = metadata_drug[metadata_drug['compound'].isin(debug_drugs)]
                print(f"DEBUG MODE: Filtered to drugs: {debug_drugs}")
            
            if exclude_drugs:
                metadata_drug = metadata_drug[~metadata_drug['compound'].isin(exclude_drugs)]
                print(f"DEBUG MODE: Excluded drugs: {exclude_drugs}")
            
            # Take only first N samples for debugging
            if debug_samples is not None and debug_samples > 0 and len(metadata_drug) > debug_samples:
                metadata_drug = metadata_drug.head(debug_samples).reset_index(drop=True)
            else:
                logger.warning("DEBUG MODE: debug_samples is None or larger than dataset size; not limiting samples.")
            
            # Ensure indices are reset after filtering
            metadata_drug = metadata_drug.reset_index(drop=True)

            print(f"DEBUG MODE: Reduced dataset size:")
            print(f"Drug metadata: {original_drug_size} → {len(metadata_drug)} samples")
            print(f"Control metadata: {original_control_size} → {len(metadata_control)} samples")
        
        # Initialize parent class with potentially filtered data
        super().__init__(
            metadata_control=metadata_control,
            metadata_drug=metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_dict=image_json_dict,
            transform=transform,
            target_size=target_size
        )
        
        # Load preprocessed drug data
        self.drug_data = self._load_drug_data(drug_data_path)
        self.drug_encoder = drug_encoder
        self.compound_lookup = self.drug_data['drug_embeddings']
        
        # Create compound mapping for quick lookup
        self._create_compound_mapping()
        
        logger.info(f"Loaded drug data for {len(self.drug_data['drug_embeddings'])} compounds")
        if debug_mode:
            logger.info(f"DEBUG MODE: Final dataset size: {len(self)} samples")

        self.fallback_smiles_dict = fallback_smiles_dict or {}
        self.enable_smiles_fallback = enable_smiles_fallback

        if enable_smiles_fallback:
            self._init_smiles_processor()

    def _init_smiles_processor(self):
        """Initialize components for on-demand SMILES processing."""
        try:
            self.rdkit_available = True
            
            # Store processing parameters from drug_data for consistency
            self.fingerprint_size = self.drug_data.get('preprocessing_params', {}).get('fingerprint_size', 1024)
            self.normalize_descriptors = self.drug_data.get('preprocessing_params', {}).get('normalize_descriptors', True)
            
            # Get normalization parameters if they exist
            if 'modality_dims' in self.drug_data:
                sample_drug = next(iter(self.drug_data['drug_embeddings'].values()))
                if 'descriptors_2d' in sample_drug and hasattr(self, 'normalization_params'):
                    self.desc_mean = self.normalization_params.get('mean')
                    self.desc_std = self.normalization_params.get('std')
        except ImportError:
            logger.warning("RDKit not available - SMILES fallback disabled")
            self.rdkit_available = False

    def _compute_smiles_embeddings(self, smiles: str) -> Dict[str, torch.Tensor]:
        """Convert SMILES to drug embeddings on-demand."""
        if not self.rdkit_available:
            logger.warning(f"Cannot process SMILES {smiles} - RDKit not available")
            return self._get_zero_embeddings()
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
            
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._get_zero_embeddings()
            
            # Compute Morgan fingerprint
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=self.fingerprint_size
            )
            fp_morgan = np.array(fp_morgan, dtype=np.float32)
            
            # Compute RDKit fingerprint
            fp_rdkit = RDKFingerprint(mol, fpSize=self.fingerprint_size)
            fp_rdkit = np.array(fp_rdkit, dtype=np.float32)
            
            # Compute 2D descriptors (same as drug_process.py)
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcExactMolWt(mol)
            ]
            descriptors_2d = np.array(descriptors, dtype=np.float32)
            
            # Apply normalization if available
            if self.normalize_descriptors and hasattr(self, 'desc_mean') and self.desc_mean is not None:
                descriptors_2d = (descriptors_2d - self.desc_mean) / (self.desc_std + 1e-8)
            
            # Convert to tensors
            result = {
                'fingerprint_morgan': torch.from_numpy(fp_morgan).float(),
                'fingerprint_rdkit': torch.from_numpy(fp_rdkit).float(),
                'descriptors_2d': torch.from_numpy(descriptors_2d).float(),
                'descriptors_3d': torch.zeros(5, dtype=torch.float32),  # No 3D info from SMILES
                'has_3d_structure': False,
                'has_2d_structure': False,  # Computed from SMILES, not actual 2D file
                'structure_source': 'SMILES_ON_DEMAND',
                'smiles': smiles
            }
            
            logger.info(f"Generated embeddings for new drug from SMILES: {smiles[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return self._get_zero_embeddings()

    def _load_drug_data(self, drug_data_path: str) -> Dict:
        """Load preprocessed drug data."""
        try:
            with open(drug_data_path, 'rb') as f:
                drug_data = pickle.load(f)
            logger.info(f"Loaded preprocessed drug data from {drug_data_path}")
            return drug_data
        except Exception as e:
            logger.error(f"Failed to load drug data from {drug_data_path}: {e}")
            raise
    
    def _create_compound_mapping(self):
        """Create mapping from compound names to drug embeddings."""
        self.compound_to_embeddings = self.drug_data['drug_embeddings']
        self.available_compounds = set(self.compound_to_embeddings.keys())
        
        # Check coverage
        required_compounds = set(self.metadata_drug['compound'].unique())
        missing_compounds = required_compounds - self.available_compounds
        
        if missing_compounds:
            logger.warning(f"Missing drug data for compounds: {missing_compounds}")
    
    def get_drug_embeddings(self, compound_name: str) -> Dict[str, torch.Tensor]:
        """Enhanced lookup with SMILES fallback."""
        # Try preprocessed lookup first
        if compound_name in self.compound_lookup:
            embeddings = self.compound_lookup[compound_name]
            result = {}
            for key, value in embeddings.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float()
                    result[key] = tensor.contiguous().clone()
                else:
                    result[key] = value
            return result
        
        # Try SMILES fallback if enabled
        elif self.enable_smiles_fallback:
            # Check if we have SMILES for this compound
            smiles = self.fallback_smiles_dict.get(compound_name)
            if smiles:
                logger.info(f"Using SMILES fallback for unknown compound: {compound_name}")
                return self._compute_smiles_embeddings(smiles)
            else:
                logger.warning(f"No SMILES available for unknown compound: {compound_name}")
                return self._get_zero_embeddings()
        
        else:
            # Original behavior - return zeros
            return self._get_zero_embeddings()

    def _get_zero_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get zero embeddings for missing compounds."""
        modality_dims = self.drug_data.get('modality_dims', {})
        
        zero_embeddings = {
            'fingerprint_morgan': torch.zeros(
                modality_dims.get('fingerprint_morgan', 1024), 
                dtype=torch.float32
            ).contiguous(),  # Ensure contiguous
            'fingerprint_rdkit': torch.zeros(
                modality_dims.get('fingerprint_rdkit', 1024), 
                dtype=torch.float32
            ).contiguous(),  # Ensure contiguous
            'descriptors_2d': torch.zeros(
                modality_dims.get('descriptors_2d', 18), 
                dtype=torch.float32
            ).contiguous(),  # Ensure contiguous
            'descriptors_3d': torch.zeros(5, dtype=torch.float32).contiguous(),
            'has_3d_structure': False,
            'has_2d_structure': False,
            'structure_source': 'NONE',
            'smiles': ''
        }
        
        return zero_embeddings
    
    def encode_drug_condition(self, compound_name: str) -> torch.Tensor:
        """
        Encode drug into condition embedding.
        
        Args:
            compound_name: Name of the compound
            
        Returns:
            Drug condition tensor
        """
        drug_embeddings = self.get_drug_embeddings(compound_name)
        
        if self.drug_encoder is not None:
            # Use trained drug encoder
            # Create a mini-batch with single item
            batch_dict = {key: value.unsqueeze(0) for key, value in drug_embeddings.items() 
                         if isinstance(value, torch.Tensor)}
            batch_dict.update({key: [value] for key, value in drug_embeddings.items() 
                              if not isinstance(value, torch.Tensor)})
            
            with torch.no_grad():
                drug_condition = self.drug_encoder(batch_dict).squeeze(0)
            return drug_condition
        else:
            # Use raw embeddings (concatenate main modalities)
            main_embeddings = [
                drug_embeddings['fingerprint_morgan'],
                drug_embeddings['descriptors_2d']
            ]
            # Add 3D descriptors if available
            if drug_embeddings['descriptors_3d'].numel() > 0:
                main_embeddings.append(drug_embeddings['descriptors_3d'])
                
            return torch.cat(main_embeddings, dim=0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Enhanced getitem that includes drug conditioning.
        
        Returns:
            Dictionary with original data plus drug conditioning
        """
        # Get original data
        sample = super().__getitem__(idx)
        
        # Get drug information
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        compound_name = treatment_sample['compound']
        
        # Add drug embeddings and condition
        drug_embeddings = self.get_drug_embeddings(compound_name)
        drug_condition = self.encode_drug_condition(compound_name)
        
        # Add to sample
        sample.update({
            'drug_embeddings': drug_embeddings,
            'drug_condition': drug_condition,
            'compound_name': compound_name
        })
        
        return sample


def create_dataloader(
    metadata_control: pd.DataFrame,
    metadata_drug: pd.DataFrame,
    gene_count_matrix: pd.DataFrame,
    image_json_path: str,
    drug_data_path: str,

    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_size: Optional[int] = 24,

    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,

    drug_encoder: Optional[torch.nn.Module] = None,
    debug_mode: bool = False,
    debug_samples: int = None,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    exclude_drugs: Optional[List[str]] = None,

    fallback_smiles_dict: Optional[Dict[str, str]] = None,
    enable_smiles_fallback: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with drug conditioning and debug options.
    
    Args:
        ... (same as before)
        debug_mode: If True, only load a subset of data for debugging
        debug_samples: Number of samples to load in debug mode
        debug_cell_lines: Specific cell lines to use for debugging
    """
    # Load image paths from JSON file
    with open(image_json_path, 'r') as f:
        image_json_dict = json.load(f)
    
    # Create dataset with debug options
    dataset = DatasetWithDrugs(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        drug_data_path=drug_data_path,

        fallback_smiles_dict=fallback_smiles_dict,
        enable_smiles_fallback=enable_smiles_fallback,

        transform=transform,
        target_size=target_size,
        
        drug_encoder=drug_encoder,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        exclude_drugs=exclude_drugs
    )
    
    if use_highly_variable_genes:
        adata = ad.AnnData(X=dataset.gene_count_matrix.T.values, 
                           obs=dataset.gene_count_matrix.T.index.to_frame(),
                           var=dataset.gene_count_matrix.T.columns.to_frame())
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        if zscore:
            sc.pp.scale(adata)
        if not normalize:
            adata.X = adata.layers['counts']
        dataset.gene_count_matrix = adata[:, hvg_genes].to_df().T

    # Enhanced collate function (same as before)
    def enhanced_collate_fn(batch):
        """Custom collate function for batch with drug data."""
        collated = {
            'control_transcriptomics': torch.stack([item['control_transcriptomics'] for item in batch]),
            'treatment_transcriptomics': torch.stack([item['treatment_transcriptomics'] for item in batch]),
            'control_images': torch.stack([item['control_images'] for item in batch]),
            'treatment_images': torch.stack([item['treatment_images'] for item in batch]),
            'compound_name': [item['compound_name'] for item in batch],
            'conditioning_info': [item['conditioning_info'] for item in batch]
        }

        # Handle drug_condition with proper error checking
        drug_conditions = []
        for item in batch:
            drug_cond = item['drug_condition']
            if drug_cond.numel() == 0:  # Empty tensor
                # Create a zero tensor with consistent shape
                drug_cond = torch.zeros(1047, dtype=torch.float32)  # Adjust size as needed
            drug_conditions.append(drug_cond)
        
        # Check if all drug conditions have the same shape
        if len(set(dc.shape for dc in drug_conditions)) == 1:
            collated['drug_condition'] = torch.stack(drug_conditions)
        else:
            # Pad to maximum length
            max_len = max(dc.shape[0] for dc in drug_conditions)
            padded_conditions = []
            for dc in drug_conditions:
                if dc.shape[0] < max_len:
                    padded = torch.zeros(max_len, dtype=torch.float32)
                    padded[:dc.shape[0]] = dc
                    padded_conditions.append(padded)
                else:
                    padded_conditions.append(dc)
            collated['drug_condition'] = torch.stack(padded_conditions)

        # Handle drug embeddings safely
        if 'drug_embeddings' in batch[0]:
            drug_keys = batch[0]['drug_embeddings'].keys()
            collated['drug_embeddings'] = {}
            for key in drug_keys:
                if isinstance(batch[0]['drug_embeddings'][key], torch.Tensor):
                    embeddings = [item['drug_embeddings'][key] for item in batch]
                    # Check if all have same shape
                    if len(set(emb.shape for emb in embeddings)) == 1:
                        collated['drug_embeddings'][key] = torch.stack(embeddings)
                    else:
                        # Handle variable shapes - pad or truncate as needed
                        collated['drug_embeddings'][key] = embeddings  # Keep as list
                else:
                    collated['drug_embeddings'][key] = [
                        item['drug_embeddings'][key] for item in batch
                    ]

        return collated
    
    # Adjust batch size and num_workers for debug mode
    if debug_mode:
        batch_size = min(batch_size, 4)
        num_workers = 0
        shuffle = False
        print(f"DEBUG MODE: Adjusted batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=enhanced_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, create_vocab_mappings(dataset)


class PrimeKGProcessor:
    """Process PrimeKG data for heterogeneous GNN."""
    
    def __init__(self, kg_csv_path: str="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/drug/PrimeKG/PrimeKG.csv"):
        self.kg_csv_path = kg_csv_path
        self.df = None
        self.node_mappings = {}
        self.edge_mappings = {}
        self.relation_types = set()
        
    def load_and_process(self, relevant_relations: List[str] = None) -> Dict:
        """Load and process the KG data."""
        logger.info(f"Loading PrimeKG from {self.kg_csv_path}")
        self.df = pd.read_csv(self.kg_csv_path)
        
        if relevant_relations is None:
            relevant_relations = ['drug_protein', 'drug_drug', 'protein_protein', 'drug_effect']
        
        # Filter for relevant relations
        self.df = self.df[self.df['relation'].isin(relevant_relations)]
        logger.info(f"Filtered to {len(self.df)} edges with relations: {relevant_relations}")
        
        # Create node mappings
        self._create_node_mappings()
        
        # Create edge mappings
        self._create_edge_mappings()
        
        return self._create_torch_geometric_data()
    
    def _create_node_mappings(self):
        """Create mappings from node IDs to indices."""
        # Collect all unique nodes - DETERMINISTIC VERSION
        x_nodes = list(zip(self.df['x_id'], self.df['x_type'], self.df['x_name']))
        y_nodes = list(zip(self.df['y_id'], self.df['y_type'], self.df['y_name']))
        
        # Use set for uniqueness, then sort with proper key function
        unique_nodes = list(set(x_nodes + y_nodes))
        
        # Sort by node_type first, then by string representation of node_id
        all_nodes = sorted(unique_nodes, key=lambda x: (x[1], str(x[0])))
        
        # Group by node type with deterministic ordering
        self.node_mappings = defaultdict(dict)
        node_type_counts = defaultdict(int)
        
        # Process nodes in sorted order by type
        nodes_by_type = defaultdict(list)
        for node_id, node_type, node_name in all_nodes:
            nodes_by_type[node_type].append((node_id, node_name))
        
        # Create mappings with consistent ordering
        for node_type in sorted(nodes_by_type.keys()):
            # Sort nodes within each type by string representation of ID
            sorted_nodes = sorted(nodes_by_type[node_type], key=lambda x: str(x[0]))
            for idx, (node_id, node_name) in enumerate(sorted_nodes):
                self.node_mappings[node_type][node_id] = {
                    'idx': idx,
                    'name': node_name,
                    'type': node_type
                }
                node_type_counts[node_type] += 1
        
        logger.info(f"Node types and counts: {dict(node_type_counts)}")
        
    def _create_edge_mappings(self):
        """Create edge type mappings."""
        self.relation_types = set(self.df['relation'].unique())
        self.edge_mappings = {rel: idx for idx, rel in enumerate(self.relation_types)}
        logger.info(f"Edge types: {list(self.relation_types)}")
        
    def _create_torch_geometric_data(self) -> Dict:
        """Create PyTorch Geometric compatible data structure."""
        # Create edge indices and types for each relation
        edge_data = {}
        
        # Sort relation types for consistent processing order
        for relation in sorted(self.relation_types):
            rel_df = self.df[self.df['relation'] == relation]
            
            # Get node type pair for this relation
            sample_row = rel_df.iloc[0]
            src_type, dst_type = sample_row['x_type'], sample_row['y_type']
            
            # Create edge indices with deterministic ordering
            edges = []
            for _, row in rel_df.iterrows():
                src_id = row['x_id']
                dst_id = row['y_id']
                
                if (src_id in self.node_mappings[src_type] and 
                    dst_id in self.node_mappings[dst_type]):
                    src_idx = self.node_mappings[src_type][src_id]['idx']
                    dst_idx = self.node_mappings[dst_type][dst_id]['idx']
                    edges.append((src_idx, dst_idx))
            
            if edges:  # Only add if we have edges
                # Sort edges for deterministic ordering
                edges = sorted(edges)
                src_indices, dst_indices = zip(*edges)
                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                edge_data[relation] = {
                    'edge_index': edge_index,
                    'src_type': src_type,
                    'dst_type': dst_type,
                    'num_edges': len(edges)
                }
        
        return {
            'edge_data': edge_data,
            'node_mappings': dict(self.node_mappings),
            'edge_mappings': self.edge_mappings,
            'num_nodes_per_type': {
                node_type: len(nodes) 
                for node_type, nodes in self.node_mappings.items()
            }
        }
    
    def get_drug_to_kg_mapping(self, drug_names: List[str]) -> Dict[str, int]:
        """Map drug names from your dataset to KG indices."""
        drug_mapping = {}
        
        if 'drug' not in self.node_mappings:
            logger.warning("No drug nodes found in KG")
            return drug_mapping
        
        # Create name-based lookup
        kg_drug_names = {
            info['name'].lower(): (drug_id, info['idx'])
            for drug_id, info in self.node_mappings['drug'].items()
        }
        
        for drug_name in drug_names:
            drug_lower = drug_name.lower()
            if drug_lower in kg_drug_names:
                drug_mapping[drug_name] = kg_drug_names[drug_lower][1]  # KG index
            else:
                logger.debug(f"Drug '{drug_name}' not found in KG")
        
        logger.info(f"Mapped {len(drug_mapping)}/{len(drug_names)} drugs to KG")
        return drug_mapping
    
    def get_gene_to_kg_mapping(self, gene_names: List[str]) -> Dict[str, int]:
        """Map gene names from RNA-seq data to KG protein indices."""
        gene_mapping = {}
        
        if 'gene/protein' not in self.node_mappings:
            logger.warning("No gene/protein nodes found in KG")
            return gene_mapping
        
        # Create name-based lookup
        kg_gene_names = {
            info['name'].upper(): (gene_id, info['idx'])
            for gene_id, info in self.node_mappings['gene/protein'].items()
        }
        
        for gene_name in gene_names:
            gene_upper = gene_name.upper()
            if gene_upper in kg_gene_names:
                gene_mapping[gene_name] = kg_gene_names[gene_upper][1]  # KG index
            else:
                logger.debug(f"Gene '{gene_name}' not found in KG")
        
        logger.info(f"Mapped {len(gene_mapping)}/{len(gene_names)} genes to KG")
        return gene_mapping
