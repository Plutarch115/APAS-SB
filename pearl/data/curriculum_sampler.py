"""
Curriculum-based data sampling for Pearl.

Implements the 5-stage curriculum training strategy described in the paper:
- Stage 1-2: Small crops (100 atoms), simple data (PDB, distillation)
- Stage 3: Medium crops (200 atoms), introduce synthetic data
- Stage 4: Large crops (500 atoms), more synthetic data
- Stage 5: Very large crops (1000 atoms), full data mixture with templates
- Final: Unlimited size, all data with full templating
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


class CurriculumStage(Enum):
    """Curriculum training stages."""
    STAGE_1 = 1  # Warmup: 100 atoms, PDB only, no templates
    STAGE_2 = 2  # Basic: 100 atoms, PDB + distillation, no templates
    STAGE_3 = 3  # Intermediate: 200 atoms, add synthetic data, simple templates
    STAGE_4 = 4  # Advanced: 500 atoms, more synthetic, complex templates
    STAGE_5 = 5  # Expert: 1000 atoms, full mixture, all templates
    FINAL = 6    # Production: unlimited, full mixture, all templates


@dataclass
class CurriculumConfig:
    """Configuration for a curriculum stage."""
    stage: CurriculumStage
    max_atoms: int
    use_pdb: bool
    use_distillation: bool
    use_synthetic: bool
    synthetic_ratio: float  # Ratio of synthetic to experimental data
    use_templates: bool
    template_complexity: str  # 'none', 'simple', 'complex', 'full'
    n_steps: int  # Number of training steps for this stage


# Default curriculum configuration from Pearl paper
DEFAULT_CURRICULUM = [
    CurriculumConfig(
        stage=CurriculumStage.STAGE_1,
        max_atoms=100,
        use_pdb=True,
        use_distillation=False,
        use_synthetic=False,
        synthetic_ratio=0.0,
        use_templates=False,
        template_complexity='none',
        n_steps=10000,
    ),
    CurriculumConfig(
        stage=CurriculumStage.STAGE_2,
        max_atoms=100,
        use_pdb=True,
        use_distillation=True,
        use_synthetic=False,
        synthetic_ratio=0.0,
        use_templates=False,
        template_complexity='none',
        n_steps=20000,
    ),
    CurriculumConfig(
        stage=CurriculumStage.STAGE_3,
        max_atoms=200,
        use_pdb=True,
        use_distillation=True,
        use_synthetic=True,
        synthetic_ratio=0.3,  # 30% synthetic
        use_templates=True,
        template_complexity='simple',
        n_steps=30000,
    ),
    CurriculumConfig(
        stage=CurriculumStage.STAGE_4,
        max_atoms=500,
        use_pdb=True,
        use_distillation=True,
        use_synthetic=True,
        synthetic_ratio=0.5,  # 50% synthetic
        use_templates=True,
        template_complexity='complex',
        n_steps=40000,
    ),
    CurriculumConfig(
        stage=CurriculumStage.STAGE_5,
        max_atoms=1000,
        use_pdb=True,
        use_distillation=True,
        use_synthetic=True,
        synthetic_ratio=0.6,  # 60% synthetic
        use_templates=True,
        template_complexity='full',
        n_steps=50000,
    ),
]


class CurriculumSampler:
    """Sample data according to curriculum training strategy."""
    
    def __init__(
        self,
        pdb_dataset: Optional[List[Dict]] = None,
        distillation_dataset: Optional[List[Dict]] = None,
        synthetic_dataset: Optional[List[Dict]] = None,
        curriculum: Optional[List[CurriculumConfig]] = None,
    ):
        """
        Args:
            pdb_dataset: Experimental PDB structures
            distillation_dataset: Distillation data from other models
            synthetic_dataset: Synthetically generated structures
            curriculum: List of curriculum configurations
        """
        self.pdb_dataset = pdb_dataset or []
        self.distillation_dataset = distillation_dataset or []
        self.synthetic_dataset = synthetic_dataset or []
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        
        # Current stage
        self.current_stage_idx = 0
        self.current_step = 0
        self.total_steps = 0
    
    def get_current_stage(self) -> CurriculumConfig:
        """Get current curriculum stage configuration."""
        return self.curriculum[self.current_stage_idx]
    
    def advance_stage(self) -> bool:
        """Advance to next curriculum stage.
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx < len(self.curriculum) - 1:
            self.current_stage_idx += 1
            self.current_step = 0
            print(f"Advanced to curriculum stage {self.current_stage_idx + 1}")
            return True
        return False
    
    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        config = self.get_current_stage()
        return self.current_step >= config.n_steps
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch according to current curriculum stage.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            List of complex dictionaries
        """
        config = self.get_current_stage()
        batch = []
        
        # Determine data sources
        available_datasets = []
        weights = []
        
        if config.use_pdb and self.pdb_dataset:
            available_datasets.append(('pdb', self.pdb_dataset))
            weights.append(1.0 - config.synthetic_ratio)
        
        if config.use_distillation and self.distillation_dataset:
            available_datasets.append(('distillation', self.distillation_dataset))
            weights.append(0.2)  # Small fraction
        
        if config.use_synthetic and self.synthetic_dataset:
            available_datasets.append(('synthetic', self.synthetic_dataset))
            weights.append(config.synthetic_ratio)
        
        if not available_datasets:
            raise ValueError("No datasets available for current curriculum stage")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample batch
        for _ in range(batch_size):
            # Choose dataset
            dataset_idx = np.random.choice(len(available_datasets), p=weights)
            dataset_name, dataset = available_datasets[dataset_idx]
            
            # Sample from dataset
            sample = dataset[np.random.randint(len(dataset))].copy()
            
            # Apply curriculum constraints
            sample['max_atoms'] = config.max_atoms
            sample['use_templates'] = config.use_templates
            sample['template_complexity'] = config.template_complexity
            sample['dataset_source'] = dataset_name
            
            batch.append(sample)
        
        self.current_step += 1
        self.total_steps += 1
        
        # Check if should advance
        if self.should_advance():
            self.advance_stage()
        
        return batch
    
    def get_statistics(self) -> Dict:
        """Get curriculum training statistics."""
        config = self.get_current_stage()
        return {
            'stage': config.stage.name,
            'stage_idx': self.current_stage_idx + 1,
            'total_stages': len(self.curriculum),
            'current_step': self.current_step,
            'stage_steps': config.n_steps,
            'total_steps': self.total_steps,
            'max_atoms': config.max_atoms,
            'synthetic_ratio': config.synthetic_ratio,
            'use_templates': config.use_templates,
            'template_complexity': config.template_complexity,
            'progress': self.current_step / config.n_steps,
        }
    
    def reset(self):
        """Reset to first curriculum stage."""
        self.current_stage_idx = 0
        self.current_step = 0
        self.total_steps = 0


class DataMixer:
    """Mix different data sources according to curriculum."""
    
    def __init__(
        self,
        pdb_weight: float = 0.4,
        synthetic_weight: float = 0.5,
        distillation_weight: float = 0.1,
    ):
        """
        Args:
            pdb_weight: Weight for PDB data
            synthetic_weight: Weight for synthetic data
            distillation_weight: Weight for distillation data
        """
        self.pdb_weight = pdb_weight
        self.synthetic_weight = synthetic_weight
        self.distillation_weight = distillation_weight
        
        # Normalize weights
        total = pdb_weight + synthetic_weight + distillation_weight
        self.pdb_weight /= total
        self.synthetic_weight /= total
        self.distillation_weight /= total
    
    def mix_datasets(
        self,
        pdb_data: List[Dict],
        synthetic_data: List[Dict],
        distillation_data: List[Dict],
        n_samples: int,
    ) -> List[Dict]:
        """Mix datasets according to weights.
        
        Args:
            pdb_data: PDB structures
            synthetic_data: Synthetic structures
            distillation_data: Distillation structures
            n_samples: Number of samples to generate
            
        Returns:
            Mixed dataset
        """
        mixed = []
        
        # Calculate number of samples from each source
        n_pdb = int(n_samples * self.pdb_weight)
        n_synthetic = int(n_samples * self.synthetic_weight)
        n_distillation = n_samples - n_pdb - n_synthetic
        
        # Sample from each dataset
        if pdb_data and n_pdb > 0:
            indices = np.random.choice(len(pdb_data), n_pdb, replace=True)
            mixed.extend([pdb_data[i] for i in indices])
        
        if synthetic_data and n_synthetic > 0:
            indices = np.random.choice(len(synthetic_data), n_synthetic, replace=True)
            mixed.extend([synthetic_data[i] for i in indices])
        
        if distillation_data and n_distillation > 0:
            indices = np.random.choice(len(distillation_data), n_distillation, replace=True)
            mixed.extend([distillation_data[i] for i in indices])
        
        # Shuffle
        np.random.shuffle(mixed)
        
        return mixed


class TemplateSelector:
    """Select templates for multi-chain templating."""
    
    def __init__(
        self,
        template_database: Optional[List[Dict]] = None,
        max_templates: int = 4,
        min_sequence_similarity: float = 0.3,
    ):
        """
        Args:
            template_database: Database of template structures
            max_templates: Maximum number of templates to use
            min_sequence_similarity: Minimum sequence similarity for templates
        """
        self.template_database = template_database or []
        self.max_templates = max_templates
        self.min_sequence_similarity = min_sequence_similarity
    
    def select_templates(
        self,
        query_sequence: str,
        complexity: str = 'simple',
    ) -> List[Dict]:
        """Select templates for query sequence.
        
        Args:
            query_sequence: Query protein sequence
            complexity: Template complexity ('simple', 'complex', 'full')
            
        Returns:
            List of template structures
        """
        if complexity == 'none' or not self.template_database:
            return []
        
        # Find similar templates
        templates = []
        for template in self.template_database:
            similarity = self._compute_similarity(
                query_sequence,
                template.get('sequence', '')
            )
            if similarity >= self.min_sequence_similarity:
                templates.append((similarity, template))
        
        # Sort by similarity
        templates.sort(key=lambda x: x[0], reverse=True)
        
        # Select top templates based on complexity
        if complexity == 'simple':
            n_templates = min(1, len(templates))
        elif complexity == 'complex':
            n_templates = min(2, len(templates))
        else:  # full
            n_templates = min(self.max_templates, len(templates))
        
        return [t[1] for t in templates[:n_templates]]
    
    def _compute_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence similarity (simple identity)."""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(len(seq1), len(seq2))

