"""
Density-Aware Training Comparison Script

This script runs a controlled experiment to validate the density-aware approach:
1. Baseline: Train with coordinates only
2. Density-aware: Train with density maps
3. Compare results to validate expected improvements

Expected improvements:
- Low-resolution (3-6Å): +20-40% better RMSD
- Medium-resolution (2-3Å): +5-10% better RMSD
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime

# Pearl imports
from pearl.data.density_map_loader import CryoEMDensityLoader, DensityMap
from pearl.models.density_generator import DifferentiableDensityGenerator
from pearl.training.density_aware_losses import DensityAwareLoss, HybridDensityCoordinateLoss

# BioPython for PDB loading
try:
    from Bio import PDB
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Install with: pip install biopython")


class SimplePearlModel(nn.Module):
    """
    Simplified Pearl model for proof-of-concept.
    
    In production, this would be the full Pearl architecture.
    For this experiment, we use a simple MLP to demonstrate the concept.
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, n_atoms, input_dim] features
        Returns:
            [batch, n_atoms, 3] predicted coordinates
        """
        features = self.encoder(x)
        coords = self.decoder(features)
        return coords


class DensityAwareExperiment:
    """
    Run controlled experiment comparing coordinate-only vs density-aware training.
    """
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load data
        print("Loading data...")
        self.structures = self._load_structures()
        print(f"Loaded {len(self.structures)} structures with density maps")
        
        # Initialize models
        print("Initializing models...")
        self.baseline_model = SimplePearlModel().to(device)
        self.density_model = SimplePearlModel().to(device)
        
        # Initialize loss functions
        self.coord_loss_fn = nn.MSELoss()
        self.density_loss_fn = DensityAwareLoss(
            coord_weight=0.3,
            density_weight=0.7,
            grid_size=32,  # Smaller grid for faster computation
            voxel_size=2.0,  # 2Å per voxel
        ).to(device)
        
        # Results storage
        self.results = {
            'baseline': [],
            'density_aware': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'device': device,
                'num_structures': len(self.structures),
            }
        }
    
    def _load_pdb_simple(self, pdb_file: Path) -> Dict:
        """Simple PDB loader using BioPython."""
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython required. Install with: pip install biopython")

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', str(pdb_file))

        # Extract all atoms
        coords = []
        atom_types = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element != 'H':  # Skip hydrogens
                            coords.append(atom.get_coord())
                            # Map element to simple index (C=0, N=1, O=2, S=3, P=4, other=5)
                            element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4}
                            atom_types.append(element_map.get(atom.element, 5))

        return {
            'coords': np.array(coords, dtype=np.float32),
            'atom_types': np.array(atom_types, dtype=np.int64),
        }

    def _load_structures(self) -> List[Dict]:
        """Load structures with both PDB and density maps."""
        # Load manifest
        manifest_file = self.data_dir / 'cryoem_manifest.json'
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        # Filter structures with both PDB and density
        structures = []
        density_loader = CryoEMDensityLoader()

        for struct in manifest['structures']:
            if struct['pdb_file'] is None:
                continue

            pdb_file = Path(struct['pdb_file'])
            map_file = Path(struct['main_map'])

            if not pdb_file.exists() or not map_file.exists():
                continue

            try:
                # Load PDB
                pdb_data = self._load_pdb_simple(pdb_file)

                # Load density map
                density_map = density_loader.load_map(map_file)

                structures.append({
                    'pdb_id': struct['pdb_id'],
                    'emdb_id': struct['emdb_id'],
                    'resolution': struct['actual_resolution'],
                    'coords': pdb_data['coords'],
                    'atom_types': pdb_data['atom_types'],
                    'density_map': density_map,
                    'description': struct['description'],
                })

                print(f"  ✓ Loaded {struct['pdb_id']} (resolution: {struct['actual_resolution']}Å, "
                      f"{len(pdb_data['coords'])} atoms)")

            except Exception as e:
                print(f"  ✗ Failed to load {struct['pdb_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return structures
    
    def _prepare_batch(self, structure: Dict) -> Dict:
        """Prepare a single structure as a batch."""
        coords = structure['coords']
        atom_types = structure['atom_types']

        # Convert to tensors
        coords_tensor = torch.from_numpy(coords).float().unsqueeze(0).to(self.device)
        atom_types_tensor = torch.from_numpy(atom_types).long().unsqueeze(0).to(self.device)

        # Create dummy input features (in real Pearl, this would be sequence embeddings)
        n_atoms = coords.shape[0]
        input_features = torch.randn(1, n_atoms, 512).to(self.device)

        # Convert density map to tensor and resize to match our grid
        density_full = structure['density_map'].to_torch(self.device)

        # Resize density map to match our grid size (32x32x32)
        # Use 3D interpolation
        density_resized = torch.nn.functional.interpolate(
            density_full.unsqueeze(0).unsqueeze(0),  # [1, 1, D, H, W]
            size=(32, 32, 32),
            mode='trilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # [32, 32, 32]

        return {
            'input_features': input_features,
            'coords': coords_tensor,
            'atom_types': atom_types_tensor,
            'density_map': density_resized.unsqueeze(0),  # [1, 32, 32, 32]
            'resolution': structure['resolution'],
        }
    
    def train_baseline(self, num_epochs: int = 50, lr: float = 1e-3):
        """Train baseline model with coordinate loss only."""
        print("\n" + "="*80)
        print("BASELINE TRAINING (Coordinate Loss Only)")
        print("="*80)
        
        optimizer = torch.optim.Adam(self.baseline_model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for struct in self.structures:
                batch = self._prepare_batch(struct)
                
                # Forward pass
                pred_coords = self.baseline_model(batch['input_features'])
                
                # Coordinate loss only
                loss = self.coord_loss_fn(pred_coords, batch['coords'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.structures)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        print("✓ Baseline training complete")
    
    def train_density_aware(self, num_epochs: int = 50, lr: float = 1e-3):
        """Train density-aware model with density loss."""
        print("\n" + "="*80)
        print("DENSITY-AWARE TRAINING (Coordinate + Density Loss)")
        print("="*80)
        
        optimizer = torch.optim.Adam(self.density_model.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_coord_loss = 0.0
            epoch_density_loss = 0.0
            
            for struct in self.structures:
                batch = self._prepare_batch(struct)
                
                # Forward pass
                pred_coords = self.density_model(batch['input_features'])
                
                # Density-aware loss
                losses = self.density_loss_fn(
                    pred_coords=pred_coords,
                    true_coords=batch['coords'],
                    atom_types=batch['atom_types'],
                    exp_density=batch['density_map'],
                )
                
                loss = losses['total_loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_coord_loss += losses['coord_rmsd'].item()
                epoch_density_loss += losses['density_loss'].item()
            
            avg_loss = epoch_loss / len(self.structures)
            avg_coord = epoch_coord_loss / len(self.structures)
            avg_density = epoch_density_loss / len(self.structures)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Total: {avg_loss:.6f}, "
                      f"Coord: {avg_coord:.6f}, Density: {avg_density:.6f}")
        
        print("✓ Density-aware training complete")
    
    def evaluate(self):
        """Evaluate both models and compare results."""
        print("\n" + "="*80)
        print("EVALUATION")
        print("="*80)
        
        self.baseline_model.eval()
        self.density_model.eval()
        
        with torch.no_grad():
            for struct in self.structures:
                batch = self._prepare_batch(struct)
                
                # Baseline predictions
                baseline_pred = self.baseline_model(batch['input_features'])
                baseline_rmsd = torch.sqrt(
                    ((baseline_pred - batch['coords']) ** 2).sum(dim=-1).mean()
                ).item()
                
                # Density-aware predictions
                density_pred = self.density_model(batch['input_features'])
                density_rmsd = torch.sqrt(
                    ((density_pred - batch['coords']) ** 2).sum(dim=-1).mean()
                ).item()
                
                # Compute improvement
                improvement = (baseline_rmsd - density_rmsd) / baseline_rmsd * 100
                
                result = {
                    'pdb_id': struct['pdb_id'],
                    'emdb_id': struct['emdb_id'],
                    'resolution': struct['resolution'],
                    'baseline_rmsd': baseline_rmsd,
                    'density_aware_rmsd': density_rmsd,
                    'improvement_percent': improvement,
                    'description': struct['description'],
                }
                
                self.results['baseline'].append(baseline_rmsd)
                self.results['density_aware'].append(density_rmsd)
                
                print(f"\n{struct['pdb_id']} ({struct['resolution']}Å):")
                print(f"  Baseline RMSD:      {baseline_rmsd:.3f} Å")
                print(f"  Density-aware RMSD: {density_rmsd:.3f} Å")
                print(f"  Improvement:        {improvement:+.1f}%")
    
    def save_results(self):
        """Save results to JSON file."""
        output_file = self.output_dir / 'density_aware_comparison_results.json'
        
        # Compute summary statistics
        baseline_mean = np.mean(self.results['baseline'])
        density_mean = np.mean(self.results['density_aware'])
        overall_improvement = (baseline_mean - density_mean) / baseline_mean * 100
        
        self.results['summary'] = {
            'baseline_mean_rmsd': float(baseline_mean),
            'density_aware_mean_rmsd': float(density_mean),
            'overall_improvement_percent': float(overall_improvement),
            'end_time': datetime.now().isoformat(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Baseline mean RMSD:      {baseline_mean:.3f} Å")
        print(f"Density-aware mean RMSD: {density_mean:.3f} Å")
        print(f"Overall improvement:     {overall_improvement:+.1f}%")
        print("="*80)
    
    def run(self, num_epochs: int = 50):
        """Run complete experiment."""
        print("\n" + "="*80)
        print("DENSITY-AWARE PEARL EXPERIMENT")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Structures: {len(self.structures)}")
        print(f"Epochs: {num_epochs}")
        
        # Train both models
        self.train_baseline(num_epochs=num_epochs)
        self.train_density_aware(num_epochs=num_epochs)
        
        # Evaluate
        self.evaluate()
        
        # Save results
        self.save_results()


def main():
    """Main entry point."""
    # Configuration
    data_dir = Path("data")
    output_dir = Path("results/density_aware_experiment")
    num_epochs = 100  # More epochs for better convergence
    
    # Run experiment
    experiment = DensityAwareExperiment(
        data_dir=data_dir,
        output_dir=output_dir,
    )
    
    experiment.run(num_epochs=num_epochs)


if __name__ == "__main__":
    main()

