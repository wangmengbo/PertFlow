import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tifffile
import json
import logging
from typing import Dict, List, Optional, Callable
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                 transform: Optional[Callable] = None):
        """
        Args:
            metadata_control: Control dataset metadata with columns ['cell_line', 'sample_id', 'json_key']
            metadata_drug: Treatment dataset metadata with columns ['cell_line', 'compound', 'timepoint', 
                          'compound_concentration_in_uM', 'sample_id', 'json_key']
            gene_count_matrix: Transcriptomics data with sample_id as columns and genes as rows
            image_json_dict: Dictionary mapping json_key to list of image paths
            transform: Optional transform to be applied on images
        """
        self.metadata_control = metadata_control
        self.metadata_drug = metadata_drug
        self.gene_count_matrix = gene_count_matrix
        self.image_json_dict = image_json_dict
        self.transform = transform
        
        for k in ['sample_id', 'cell_line', 'json_key', 'compound',]:
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
        
        images = []
        for path in image_paths:
            try:
                img = tifffile.imread(path)
                # Ensure 2D image (H, W)
                if img.ndim > 2:
                    img = img.squeeze()
                images.append(img)
            except Exception as e:
                logger.info(f"Error loading image {path}: {e}")
                raise
        
        # Stack images along the channel dimension to form (C, H, W)
        images = np.stack(images, axis=0).astype(np.float32)
        
        # Apply transform if provided
        if self.transform:
            images = self.transform(images)
            
        return images
    
    def get_transcriptomics_data(self, sample_id: str) -> np.ndarray:
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
        
        return self.gene_count_matrix[sample_id].values.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired sample (control + treatment) with conditioning information.
        
        Args:
            idx: Index of the treatment sample
            
        Returns:
            Dictionary containing paired data and conditioning information
        """
        # Get treatment sample metadata
        logger.info(f"treatment_sample['sample_id']=\"{self.filtered_drug_metadata.iloc[idx]['sample_id']}\"")
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        cell_line = treatment_sample['cell_line']
        
        # Sample a random control sample with the same cell_line
        control_samples = self.control_grouped.get_group(cell_line)
        control_sample = control_samples.sample(n=1).iloc[0]
        
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
        
        # Return paired data as tensors
        return {
            'control_transcriptomics': torch.tensor(control_transcriptomics),
            'treatment_transcriptomics': torch.tensor(treatment_transcriptomics),
            'control_images': torch.tensor(treatment_images),
            'treatment_images': torch.tensor(control_images),
            'conditioning_info': conditioning_info
        }

def create_dataloader(metadata_control: pd.DataFrame,
                     metadata_drug: pd.DataFrame,
                     gene_count_matrix: pd.DataFrame,
                     image_json_path: str,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     transform: Optional[Callable] = None) -> DataLoader:
    """
    Create a DataLoader for the paired histology and transcriptomics dataset.
    
    Args:
        metadata_control: Control dataset metadata
        metadata_drug: Treatment dataset metadata
        gene_count_matrix: Transcriptomics data
        image_json_path: Path to JSON file containing image paths
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional image transform
        
    Returns:
        PyTorch DataLoader
    """
    # Load image paths from JSON file
    with open(image_json_path, 'r') as f:
        image_json_dict = json.load(f)
    
    # Create dataset
    dataset = HistologyTranscriptomicsDataset(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        transform=transform
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # For faster GPU transfer
    )
    
    return dataloader


def image_transform(images):
    # Custom transform for multi-channel images
    # images shape: (channels, height, width)
    # Add any preprocessing here (normalization, resizing, etc.)
    return images / 255.0  # Simple normalization


if __name__ == '__main__':

    # Load your data
    metadata_control = pd.read_csv('/depot/natallah/data/Mengbo/HnE_RNA/PerturbGeneFlow/data/processed_data/metadata_control.csv')
    metadata_drug = pd.read_csv('/depot/natallah/data/Mengbo/HnE_RNA/PerturbGeneFlow/data/processed_data//metadata_drug.csv')
    gene_count_matrix = pd.read_parquet('/depot/natallah/data/Mengbo/HnE_RNA/PerturbGeneFlow/data/processed_data//GDPx1x2_gene_counts.parquet')  # Genes as rows, samples as columns
    image_json_path = "/depot/natallah/data/Mengbo/HnE_RNA/PerturbGeneFlow/data/processed_data//image_paths.json"

    # Create DataLoader
    dataloader = create_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_path=image_json_path,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        transform=image_transform
    )

    # Training loop example
    for batch in dataloader:
        control_transcriptomics = batch['control_transcriptomics']  # Shape: (batch_size, num_genes)
        treatment_transcriptomics = batch['treatment_transcriptomics']  # Shape: (batch_size, num_genes)
        control_images = batch['control_images']  # Shape: (batch_size, channels, height, width)
        treatment_images = batch['treatment_images']  # Shape: (batch_size, channels, height, width)
        conditioning_info = batch['conditioning_info']  # Dictionary with conditioning variables
        
        # Your model training code here
        # model_output = model(control_transcriptomics, control_images, conditioning_info)
        break

