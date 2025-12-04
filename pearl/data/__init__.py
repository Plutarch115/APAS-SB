"""
Data loading and processing for Pearl.

This module provides:
- PDB data loading and parsing
- Synthetic data generation
- Feature extraction and preprocessing
- Curriculum-based data sampling
"""

from .pdb_loader import PDBDataset, PDBDataLoader
from .synthetic_generator import (
    SyntheticDataGenerator,
    VirtualLigandLibrary,
    PhysicsBasedDocker,
)
from .preprocessing import (
    ProteinFeaturizer,
    LigandFeaturizer,
    ComplexPreprocessor,
    CroppingStrategy
)
from .curriculum_sampler import CurriculumSampler, CurriculumStage, CurriculumConfig

__all__ = [
    'PDBDataset',
    'PDBDataLoader',
    'SyntheticDataGenerator',
    'VirtualLigandLibrary',
    'PhysicsBasedDocker',
    'ProteinFeaturizer',
    'LigandFeaturizer',
    'ComplexPreprocessor',
    'CroppingStrategy',
    'CurriculumSampler',
    'CurriculumStage',
    'CurriculumConfig',
]

