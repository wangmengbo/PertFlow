import warnings
warnings.filterwarnings('ignore', message='.*MorganGenerator.*')
warnings.filterwarnings('ignore', message='.*pin_memory.*')
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D, rdMolDescriptors, AllChem, RDKFingerprint
import os
from typing import Optional, Dict, List, Union, Tuple
import pickle
import h5py
from tqdm import tqdm
from datetime import datetime


class PubChemDrugDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/home/wang4887/HnE_RNA/PerturbGeneFlow/data/drug/PubChem",
        representation: str = "fingerprint",
        fingerprint_type: str = "morgan",
        fingerprint_size: int = 2048,
        radius: int = 2,
        use_3d_sdf: bool = False,
        normalize_descriptors: bool = True,
        include_basic_props: bool = True,
        filter_valid_smiles: bool = True,
        debug_mode: bool = False,           # NEW
        debug_samples: int = 50             # NEW
    ):
        # Store all parameters
        self.data_dir = data_dir
        self.representation = representation
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.radius = radius
        self.use_3d_sdf = use_3d_sdf
        self.normalize_descriptors = normalize_descriptors
        self.include_basic_props = include_basic_props
        self.debug_mode = debug_mode         # NEW
        self.debug_samples = debug_samples   # NEW
        
        # Load data files
        self.complete_data = pd.read_csv(os.path.join(data_dir, "complete_drug_data.csv"))
        self.basic_info = pd.read_csv(os.path.join(data_dir, "drug_basic_info.csv"))
        self.sdf_index = pd.read_csv(os.path.join(data_dir, "sdf_file_index.csv"))
        
        # Merge datasets
        self.data = self._merge_datasets()
        
        # Apply debug mode early if requested
        if self.debug_mode:
            original_size = len(self.data)
            self.data = self.data.head(self.debug_samples).reset_index(drop=True)
            print(f"🐛 DEBUG MODE: Using {len(self.data)}/{original_size} samples")
        
        # Filter valid SMILES if requested
        if filter_valid_smiles:
            self.data = self._filter_valid_smiles()
        
        # Precompute molecular representations
        self._precompute_representations()
        
        print(f"Loaded {len(self.data)} drugs with {self.representation} representation")
    
    def _merge_datasets(self) -> pd.DataFrame:
        """Merge all data sources into a single DataFrame."""
        # Start with complete data
        merged = self.complete_data.copy()
        
        # Add SDF file paths
        merged = merged.merge(
            self.sdf_index[['cid', 'sdf_3d_file', 'sdf_2d_file']], 
            on='cid', 
            how='left'
        )
        
        return merged
    
    def _filter_valid_smiles(self) -> pd.DataFrame:
        """Filter out drugs without valid SMILES strings."""
        valid_smiles = []
        for idx, row in self.data.iterrows():
            smiles = row.get('canonical_smiles') or row.get('isomeric_smiles')
            if smiles and pd.notna(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(idx)
        
        print(f"Filtered from {len(self.data)} to {len(valid_smiles)} drugs with valid SMILES")
        return self.data.iloc[valid_smiles].reset_index(drop=True)
    
    def _get_mol_from_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit molecule object."""
        if pd.isna(smiles) or not smiles:
            return None
        return Chem.MolFromSmiles(smiles)
    
    def _get_mol_from_sdf(self, sdf_path: str) -> Optional[Chem.Mol]:
        """Load molecule from SDF file."""
        if pd.isna(sdf_path) or not os.path.exists(sdf_path):
            return None
        try:
            supplier = Chem.SDMolSupplier(sdf_path)
            mol = next(supplier)
            return mol
        except:
            return None
    
    def _compute_molecular_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Compute molecular descriptors using RDKit."""
        if mol is None:
            return np.array([])
        
        descriptors = []
        
        # Basic molecular properties
        if self.include_basic_props:
            descriptors.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),  # FIXED: Use rdMolDescriptors
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol)
            ])
        
        # Additional descriptors
        descriptors.extend([
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),  # This is already correct
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcExactMolWt(mol)
        ])
        
        return np.array(descriptors, dtype=np.float32)

    
    def _compute_fingerprint(self, mol, fp_type='morgan'):
        """Enhanced fingerprint computation with type selection."""
        if mol is None:
            return np.zeros(self.fingerprint_size, dtype=np.float32)
        
        if fp_type == "morgan":
            # FIXED: Use new MorganGenerator instead of deprecated function
            try:
                # Use the new MorganGenerator API
                generator = rdMolDescriptors.GetMorganGenerator(radius=self.radius)
                fp = generator.GetFingerprintAsNumPy(mol, nBits=self.fingerprint_size)
            except (ImportError, AttributeError):
                # Fallback to old method if new API not available

                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.fingerprint_size
                )
                fp = np.array(fp, dtype=np.float32)
            
        elif fp_type == "rdkit":
            fp = RDKFingerprint(mol, fpSize=self.fingerprint_size)
            fp = np.array(fp, dtype=np.float32)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        return fp

    
    def _precompute_representations(self):
        """Precompute molecular representations for all drugs."""
        self.representations = []
        self.drug_info = []
        
        for idx, row in self.data.iterrows():
            # Get molecule object
            mol = None
            
            # Try to get molecule from SDF file first (if requested and available)
            if self.use_3d_sdf and pd.notna(row.get('sdf_3d_file')):
                mol = self._get_mol_from_sdf(row['sdf_3d_file'])
            
            # Fall back to 2D SDF
            if mol is None and pd.notna(row.get('sdf_2d_file')):
                mol = self._get_mol_from_sdf(row['sdf_2d_file'])
            
            # Fall back to SMILES
            if mol is None:
                smiles = row.get('canonical_smiles') or row.get('isomeric_smiles')
                if smiles:
                    mol = self._get_mol_from_smiles(smiles)
            
            # Compute representation
            if self.representation == "smiles":
                # For SMILES, we'll return the string (to be tokenized later)
                smiles = row.get('canonical_smiles') or row.get('isomeric_smiles') or ""
                self.representations.append(smiles)
            
            elif self.representation == "descriptors":
                desc = self._compute_molecular_descriptors(mol)
                self.representations.append(desc)
            
            elif self.representation == "fingerprint":
                fp = self._compute_fingerprint(mol)
                self.representations.append(fp)
            
            elif self.representation == "combined":
                desc = self._compute_molecular_descriptors(mol)
                fp = self._compute_fingerprint(mol)
                combined = np.concatenate([desc, fp]) if len(desc) > 0 else fp
                self.representations.append(combined)
            
            # Store drug metadata
            self.drug_info.append({
                'name': row.get('original_name', ''),
                'cid': row.get('cid', 0),
                'molecular_formula': row.get('molecular_formula', ''),
                'molecular_weight': row.get('molecular_weight', 0.0),
                'iupac_name': row.get('iupac_name', ''),
                'inchikey': row.get('inchikey', '')
            })
        
        # Normalize descriptors if requested
        if self.normalize_descriptors and self.representation in ["descriptors", "combined"]:
            self._normalize_representations()
    
    def _normalize_representations(self):
        """Normalize molecular descriptors."""
        if self.representation == "smiles":
            return
        
        # Convert to array for normalization
        repr_array = np.array([r for r in self.representations if len(r) > 0])
        
        if len(repr_array) == 0:
            return
        
        # Compute mean and std, avoiding division by zero
        mean = np.mean(repr_array, axis=0)
        std = np.std(repr_array, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        
        # Normalize
        normalized = []
        for repr_vec in self.representations:
            if len(repr_vec) > 0:
                norm_vec = (repr_vec - mean) / std
                normalized.append(norm_vec)
            else:
                normalized.append(repr_vec)
        
        self.representations = normalized
        self.normalization_params = {'mean': mean, 'std': std}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        """Get a single data sample."""
        representation = self.representations[idx]
        drug_info = self.drug_info[idx]
        
        # Convert to tensor if not SMILES
        if self.representation != "smiles":
            if len(representation) > 0:
                representation = torch.tensor(representation, dtype=torch.float32)
            else:
                # Return zero vector if representation failed
                if self.representation == "fingerprint":
                    representation = torch.zeros(self.fingerprint_size, dtype=torch.float32)
                else:
                    representation = torch.zeros(10, dtype=torch.float32)  # Fallback size
        
        return {
            'representation': representation,
            'drug_name': drug_info['name'],
            'cid': drug_info['cid'],
            'molecular_formula': drug_info['molecular_formula'],
            'molecular_weight': drug_info['molecular_weight'],
            'iupac_name': drug_info['iupac_name'],
            'inchikey': drug_info['inchikey']
        }
    
    def get_representation_dim(self) -> int:
        """Get the dimensionality of the molecular representation."""
        if len(self.representations) == 0:
            return 0
        
        if self.representation == "smiles":
            return -1  # Variable length, needs tokenization
        
        sample_repr = self.representations[0]
        return len(sample_repr) if len(sample_repr) > 0 else 0


def create_drug_dataloader(
    data_dir: str,
    batch_size: int = 32,
    representation: str = "fingerprint",
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for PubChem drug data.
    
    Args:
        data_dir: Path to PubChem data directory
        batch_size: Batch size for training
        representation: Type of molecular representation
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for PubChemDrugDataset
    
    Returns:
        DataLoader object
    """
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length representations."""
        if batch[0]['representation'] == "smiles":
            # For SMILES, return as list of strings
            return {
                'representation': [item['representation'] for item in batch],
                'drug_name': [item['drug_name'] for item in batch],
                'cid': torch.tensor([item['cid'] for item in batch]),
                'molecular_weight': torch.tensor([item['molecular_weight'] for item in batch]),
                'iupac_name': [item['iupac_name'] for item in batch],
                'inchikey': [item['inchikey'] for item in batch]
            }
        else:
            # For numerical representations, stack as tensors
            return {
                'representation': torch.stack([item['representation'] for item in batch]),
                'drug_name': [item['drug_name'] for item in batch],
                'cid': torch.tensor([item['cid'] for item in batch]),
                'molecular_weight': torch.tensor([item['molecular_weight'] for item in batch]),
                'iupac_name': [item['iupac_name'] for item in batch],
                'inchikey': [item['inchikey'] for item in batch]
            }
    
    dataset = PubChemDrugDataset(data_dir=data_dir, representation=representation, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


class MultiModalDrugDataset(PubChemDrugDataset):
    """Enhanced dataset that provides multiple representations simultaneously."""
    
    def __init__(self, data_dir: str, debug_mode: bool = False, debug_samples: int = 50, **kwargs):
        # Store debug parameters
        self.debug_mode = debug_mode
        self.debug_samples = debug_samples
        
        # Remove representation parameter since we compute all
        kwargs.pop('representation', None)
        
        # **FIXED: Pass debug parameters to parent class**
        super().__init__(
            data_dir=data_dir, 
            representation="combined",
            debug_mode=debug_mode,      # Pass debug mode to parent
            debug_samples=debug_samples, # Pass debug samples to parent
            **kwargs
        )
        
        # Remove the _apply_debug_mode() call since parent already handles it
        # Precompute all representation types
        self._precompute_all_representations()

    def _apply_debug_mode(self):
        """Reduce dataset size for debugging purposes."""
        original_size = len(self.data)
        
        if self.debug_samples < original_size:
            # Take first N samples (or random samples if shuffled)
            if hasattr(self, 'shuffle') and self.shuffle:
                # Random sampling for better representation
                import random
                indices = random.sample(range(original_size), self.debug_samples)
                self.data = self.data.iloc[indices].reset_index(drop=True)
            else:
                # Take first N samples
                self.data = self.data.head(self.debug_samples).reset_index(drop=True)
            
            print(f"🐛 DEBUG MODE: Reduced dataset from {original_size} to {len(self.data)} samples")
        else:
            print(f"🐛 DEBUG MODE: Requested {self.debug_samples} samples, but dataset only has {original_size}")
    
    def _get_structure_source(self, mol_3d, mol_2d, mol_smiles):
        """Determine the primary structure source."""
        if mol_3d is not None:
            return "3D_SDF"
        elif mol_2d is not None:
            return "2D_SDF"
        elif mol_smiles is not None:
            return "SMILES"
        else:
            return "NONE"
    
    def _compute_3d_descriptors(self, mol_3d):
        """Compute 3D-specific descriptors."""
        if mol_3d is None:
            return np.array([])
        
        try:
            descriptors_3d = [
                rdMolDescriptors.CalcSpherocityIndex(mol_3d),
                rdMolDescriptors.CalcAsphericity(mol_3d),
                rdMolDescriptors.CalcEccentricity(mol_3d),
                rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
                rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
                # Add more 3D descriptors as needed
            ]
            return np.array(descriptors_3d, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Could not compute 3D descriptors: {e}")
            return np.array([])
    
    def _precompute_all_representations(self):
        """Compute all possible representations for each drug."""
        self.multi_representations = []
        
        for idx, row in self.data.iterrows():
            # Get molecules from different sources
            mol_3d = None
            mol_2d = None
            mol_smiles = None
            
            # 3D structure
            if pd.notna(row.get('sdf_3d_file')):
                mol_3d = self._get_mol_from_sdf(row['sdf_3d_file'])
            
            # 2D structure  
            if pd.notna(row.get('sdf_2d_file')):
                mol_2d = self._get_mol_from_sdf(row['sdf_2d_file'])
            
            # SMILES
            smiles = row.get('canonical_smiles') or row.get('isomeric_smiles')
            if smiles:
                mol_smiles = self._get_mol_from_smiles(smiles)
            
            # Use best available molecule for computations
            best_mol = mol_3d or mol_2d or mol_smiles
            
            representations = {
                'smiles': smiles or "",
                'fingerprint_morgan': self._compute_fingerprint(best_mol, fp_type='morgan'),
                'fingerprint_rdkit': self._compute_fingerprint(best_mol, fp_type='rdkit'),
                'descriptors_2d': self._compute_molecular_descriptors(mol_2d or best_mol),
                'descriptors_3d': self._compute_3d_descriptors(mol_3d) if mol_3d else np.array([]),
                'has_3d_structure': mol_3d is not None,
                'has_2d_structure': mol_2d is not None,
                'structure_source': self._get_structure_source(mol_3d, mol_2d, mol_smiles)
            }
            
            self.multi_representations.append(representations)
    
    def _compute_fingerprint(self, mol, fp_type='morgan'):
        """Enhanced fingerprint computation with type selection."""
        if mol is None:
            return np.zeros(self.fingerprint_size, dtype=np.float32)
        
        if fp_type == "morgan":
            # Use old method but suppress warnings
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.fingerprint_size
            )
            return np.array(fp, dtype=np.float32)
        
        elif fp_type == "rdkit":
            fp = RDKFingerprint(mol, fpSize=self.fingerprint_size)
            return np.array(fp, dtype=np.float32)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
    
    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, str, int, float, bool]]:
        """Return all available representations for a single drug."""
        # Get basic drug info from parent class
        base_item = super().__getitem__(idx)
        
        # Get multi-modal representations
        multi_repr = self.multi_representations[idx]
        
        # Convert numpy arrays to tensors
        processed_repr = {}
        for key, value in multi_repr.items():
            if isinstance(value, np.ndarray) and len(value) > 0:
                processed_repr[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, np.ndarray):
                # Empty array
                processed_repr[key] = torch.tensor([], dtype=torch.float32)
            else:
                # Keep strings, bools, etc. as-is
                processed_repr[key] = value
        
        # Update base item with multi-modal representations
        base_item.update(processed_repr)
        
        return base_item
    
    def get_modality_dims(self):
        """Return dimensions of each modality."""
        if len(self.multi_representations) == 0:
            return {}
        
        sample = self.multi_representations[0]
        dims = {}
        
        for key, value in sample.items():
            if isinstance(value, np.ndarray) and len(value) > 0:
                dims[key] = len(value)
            elif key.startswith('fingerprint_'):
                dims[key] = self.fingerprint_size
            elif key == 'smiles':
                dims[key] = -1  # Variable length
            elif key.startswith('has_') or key == 'structure_source':
                dims[key] = 'categorical'  # Better than None
            else:
                dims[key] = 'metadata'
        
        return dims
    
    def get_representation_statistics(self):
        """Get statistics about available representations across the dataset."""
        stats = {
            'total_drugs': len(self.multi_representations),
            'has_3d_structure': 0,
            'has_2d_structure': 0,
            'has_smiles': 0,
            'structure_sources': {'3D_SDF': 0, '2D_SDF': 0, 'SMILES': 0, 'NONE': 0}
        }
        
        for repr_dict in self.multi_representations:
            if repr_dict['has_3d_structure']:
                stats['has_3d_structure'] += 1
            if repr_dict['has_2d_structure']:
                stats['has_2d_structure'] += 1
            if repr_dict['smiles']:
                stats['has_smiles'] += 1
            
            source = repr_dict['structure_source']
            if source in stats['structure_sources']:
                stats['structure_sources'][source] += 1
        
        return stats


def create_multimodal_drug_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    fingerprint_size: int = 2048,
    normalize_descriptors: bool = True,
    use_3d_sdf: bool = True,
    pad_missing_modalities: bool = True,
    debug_mode: bool = False,          # NEW: Enable debug mode
    debug_samples: int = 50,           # NEW: Number of samples in debug mode
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for multi-modal PubChem drug data.
    
    Args:
        data_dir: Path to PubChem data directory
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        fingerprint_size: Size of fingerprint vectors
        normalize_descriptors: Whether to normalize molecular descriptors
        use_3d_sdf: Whether to prefer 3D SDF files when available
        pad_missing_modalities: Whether to pad missing modalities with zeros
        debug_mode: If True, only load a subset of data for debugging
        debug_samples: Number of samples to load in debug mode
        **kwargs: Additional arguments for MultiModalDrugDataset
    
    Returns:
        DataLoader object with multi-modal batches
    """
    
    def multimodal_collate_fn(batch):
        """Custom collate function for multi-modal drug data."""
        batch_size = len(batch)
        
        # Initialize batch dictionary
        collated_batch = {
            'drug_name': [item['drug_name'] for item in batch],
            'cid': torch.tensor([item['cid'] for item in batch], dtype=torch.long),
            'molecular_weight': torch.tensor([item['molecular_weight'] for item in batch], dtype=torch.float32),
            'iupac_name': [item['iupac_name'] for item in batch],
            'inchikey': [item['inchikey'] for item in batch],
        }
        
        # Handle SMILES (variable length strings)
        collated_batch['smiles'] = [item['smiles'] for item in batch]
        
        # Handle fingerprints
        fingerprint_keys = ['fingerprint_morgan', 'fingerprint_rdkit']
        for fp_key in fingerprint_keys:
            if fp_key in batch[0]:
                fp_tensors = []
                for item in batch:
                    fp = item[fp_key]
                    if isinstance(fp, torch.Tensor) and fp.numel() > 0:
                        fp_tensors.append(fp)
                    elif pad_missing_modalities:
                        # Pad with zeros if missing
                        fp_tensors.append(torch.zeros(fingerprint_size, dtype=torch.float32))
                    else:
                        fp_tensors.append(torch.tensor([], dtype=torch.float32))
                
                if pad_missing_modalities:
                    collated_batch[fp_key] = torch.stack(fp_tensors)
                else:
                    collated_batch[fp_key] = fp_tensors
        
        # Handle descriptors
        descriptor_keys = ['descriptors_2d', 'descriptors_3d']
        for desc_key in descriptor_keys:
            if desc_key in batch[0]:
                desc_tensors = []
                desc_dims = []
                
                # First pass: collect dimensions
                for item in batch:
                    desc = item[desc_key]
                    if isinstance(desc, torch.Tensor) and desc.numel() > 0:
                        desc_dims.append(desc.shape[0])
                    else:
                        desc_dims.append(0)
                
                # Determine max dimension for padding
                max_dim = max(desc_dims) if desc_dims else 0
                
                # Second pass: pad to max dimension
                for i, item in enumerate(batch):
                    desc = item[desc_key]
                    if isinstance(desc, torch.Tensor) and desc.numel() > 0:
                        if pad_missing_modalities and desc.shape[0] < max_dim:
                            # Pad to max dimension
                            padded_desc = torch.zeros(max_dim, dtype=torch.float32)
                            padded_desc[:desc.shape[0]] = desc
                            desc_tensors.append(padded_desc)
                        else:
                            desc_tensors.append(desc)
                    elif pad_missing_modalities:
                        # Missing descriptor - pad with zeros
                        desc_tensors.append(torch.zeros(max_dim, dtype=torch.float32))
                    else:
                        desc_tensors.append(torch.tensor([], dtype=torch.float32))
                
                if pad_missing_modalities and max_dim > 0:
                    collated_batch[desc_key] = torch.stack(desc_tensors)
                    # Create mask for valid descriptors
                    mask = torch.tensor([dim > 0 for dim in desc_dims], dtype=torch.bool)
                    collated_batch[f'{desc_key}_mask'] = mask
                else:
                    collated_batch[desc_key] = desc_tensors
        
        # Handle boolean flags
        bool_keys = ['has_3d_structure', 'has_2d_structure']
        for bool_key in bool_keys:
            if bool_key in batch[0]:
                collated_batch[bool_key] = torch.tensor([item[bool_key] for item in batch], dtype=torch.bool)
        
        # Handle structure source
        if 'structure_source' in batch[0]:
            collated_batch['structure_source'] = [item['structure_source'] for item in batch]
        
        return collated_batch
        
    # Pass debug parameters to dataset
    dataset = MultiModalDrugDataset(
        data_dir=data_dir,
        fingerprint_size=fingerprint_size,
        normalize_descriptors=normalize_descriptors,
        use_3d_sdf=use_3d_sdf,
        debug_mode=debug_mode,           # NEW
        debug_samples=debug_samples,     # NEW
        **kwargs
    )
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=torch.cuda.is_available()  # Only use pin_memory if GPU available
    )
    
    return dataloader


def preprocess_and_save_drug_data(
    data_dir: str,
    output_path: str,
    fingerprint_size: int = 1024,
    use_3d_sdf: bool = True,
    normalize_descriptors: bool = True,
    drug_name_mapping: Optional[Dict[str, str]] = None,
    debug_mode: bool = False,
    debug_samples: int = 50,
    include_metadata: bool = True
):
    """
    Preprocess all drug data and save to single file for fast loading.
    """
    import pickle
    from tqdm import tqdm
    
    print(f"🧬 Preprocessing drug data from {data_dir}")
    
    dataset = MultiModalDrugDataset(
        data_dir=data_dir,
        fingerprint_size=fingerprint_size,
        use_3d_sdf=use_3d_sdf,
        normalize_descriptors=normalize_descriptors,
        debug_mode=debug_mode,
        debug_samples=debug_samples
    )
    
    # Create comprehensive drug lookup
    drug_lookup = {}
    compound_metadata = {}
    
    print(f"Processing {len(dataset)} drugs...")
    
    for idx in tqdm(range(len(dataset))):
        drug_data = dataset[idx]
        compound_name = drug_data['drug_name'] if drug_name_mapping is None else drug_name_mapping.get(drug_data['drug_name'], drug_data['drug_name'])
        
        # Store all drug embeddings (converted to numpy for serialization)
        drug_embeddings = {}
        for key, value in drug_data.items():
            if key in ['fingerprint_morgan', 'fingerprint_rdkit', 'descriptors_2d', 'descriptors_3d']:
                if isinstance(value, torch.Tensor):
                    drug_embeddings[key] = value.numpy()
                else:
                    drug_embeddings[key] = value
            elif key in ['has_3d_structure', 'has_2d_structure', 'structure_source', 'smiles']:
                drug_embeddings[key] = value
        
        # Store metadata separately for easy access
        if include_metadata:
            compound_metadata[compound_name] = {
                'cid': drug_data['cid'],
                'molecular_weight': drug_data['molecular_weight'],
                'iupac_name': drug_data['iupac_name'],
                'inchikey': drug_data['inchikey']
            }
        
        drug_lookup[compound_name] = drug_embeddings
    
    # Comprehensive save data structure
    save_data = {
        'version': '1.0',
        'timestamp': pd.Timestamp.now().isoformat(),
        'drug_embeddings': drug_lookup,
        'compound_metadata': compound_metadata,
        'modality_dims': dataset.get_modality_dims(),
        'dataset_stats': dataset.get_representation_statistics(),
        'preprocessing_params': {
            'fingerprint_size': fingerprint_size,
            'use_3d_sdf': use_3d_sdf,
            'normalize_descriptors': normalize_descriptors,
            'debug_mode': debug_mode,
            'total_compounds': len(drug_lookup)
        }
    }
    
    # Save with compression for efficiency
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved {len(drug_lookup)} drugs to {output_path}")
    print(f"📁 File size: {file_size_mb:.2f} MB")
    print(f"📊 Average size per drug: {file_size_mb/len(drug_lookup):.3f} MB")
    
    return save_data


def load_preprocessed_drug_data(file_path: str, file_format: str = "pickle"):
    """Load preprocessed drug data from file."""
    import pickle
    import h5py
    
    if file_format == "pickle":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_format == "h5":
        # For H5, you'd implement loading logic here
        # This is more complex due to H5 structure
        pass
    else:
        raise ValueError(f"Unsupported format: {file_format}")


# Usage example
if __name__ == "__main__":

    data_dir = "/home/wang4887/HnE_RNA/PerturbGeneFlow/data/drug/PubChem"

    ########### multi ############
    # Create multi-modal dataloader
    multimodal_loader = create_multimodal_drug_dataloader(
        data_dir=data_dir,
        batch_size=4,
        fingerprint_size=1024,
        use_3d_sdf=True,
        pad_missing_modalities=True,
        debug_mode=True,
        debug_samples=20
    )

    # Test the dataloader
    print("Testing multi-modal dataloader:")
    for batch in multimodal_loader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Morgan fingerprint shape: {batch['fingerprint_morgan'].shape}")
        print(f"2D descriptors shape: {batch['descriptors_2d'].shape}")
        if 'descriptors_3d' in batch:
            print(f"3D descriptors shape: {batch['descriptors_3d'].shape}")
        print(f"Has 3D structure: {batch['has_3d_structure'].sum().item()}/{len(batch['has_3d_structure'])} drugs")
        break

    # Get modality dimensions for model design
    print(f"Modality dimensions: {multimodal_loader.dataset.get_modality_dims()}")

    # data_dir = "/depot/natallah/data/Mengbo/HnE_RNA/PerturbGeneFlow/data/drug/PubChem"
    
    # ########### multi ############
    # # Create multi-modal dataloader
    # multimodal_loader = create_multimodal_drug_dataloader(
    #     data_dir=data_dir,
    #     batch_size=8,
    #     fingerprint_size=1024,
    #     use_3d_sdf=True,
    #     pad_missing_modalities=True
    # )
    
    # # Test the dataloader
    # print("Testing multi-modal dataloader:")
    # for batch in multimodal_loader:
    #     print(f"Batch keys: {list(batch.keys())}")
    #     print(f"Morgan fingerprint shape: {batch['fingerprint_morgan'].shape}")
    #     print(f"2D descriptors shape: {batch['descriptors_2d'].shape}")
    #     if 'descriptors_3d' in batch:
    #         print(f"3D descriptors shape: {batch['descriptors_3d'].shape}")
    #     print(f"Has 3D structure: {batch['has_3d_structure'].sum().item()}/{len(batch['has_3d_structure'])} drugs")
    #     break
    
    # # Get modality dimensions for model design
    # print(f"Modality dimensions: {multimodal_loader.dataset.get_modality_dims()}")

    
    # ########### single ############
    # # Create dataloaders for different representations
    # fingerprint_loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     batch_size=16,
    #     representation="fingerprint",
    #     fingerprint_type="morgan",
    #     fingerprint_size=1024
    # )
    
    # descriptor_loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     batch_size=16,
    #     representation="descriptors",
    #     normalize_descriptors=True
    # )
    
    # combined_loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     batch_size=16,
    #     representation="combined"
    # )
    
    # # For neural networks with fingerprints
    # loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     representation="fingerprint",
    #     fingerprint_type="morgan",
    #     fingerprint_size=2048,
    #     batch_size=32
    # )

    # # Test the dataloader
    # print("Testing fingerprint representation:")
    # for batch in fingerprint_loader:
    #     print(f"Batch shape: {batch['representation'].shape}")
    #     print(f"Drug names: {batch['drug_name'][:3]}")
    #     print(f"Representation dim: {fingerprint_loader.dataset.get_representation_dim()}")
    #     break
    
    # print("\nTesting descriptor representation:")
    # for batch in descriptor_loader:
    #     print(f"Batch shape: {batch['representation'].shape}")
    #     print(f"Representation dim: {descriptor_loader.dataset.get_representation_dim()}")
    #     break

    # # For transformer models with SMILES
    # smiles_loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     representation="smiles",
    #     batch_size=16
    # )

    # # For traditional ML with descriptors
    # desc_loader = create_drug_dataloader(
    #     data_dir=data_dir,
    #     representation="descriptors",
    #     normalize_descriptors=True
    # )

