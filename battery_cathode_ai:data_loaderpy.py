"""
Data Loader Module supporting Graph and Descriptor modes.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from graph_builder import CIFData # Import the graph builder class

class BenchmarkSetLoader:
    """Loads the benchmark JSON file."""
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data['benchmark_materials']

class GraphDataset(Dataset):
    """PyTorch Dataset for crystal graphs."""
    def __init__(self, cif_paths, benchmark_data_map):
        self.cif_paths = cif_paths
        self.benchmark_data_map = benchmark_data_map
        self.cif_data = CIFData(cif_paths)

    def __len__(self):
        return len(self.cif_paths)

    def __getitem__(self, idx):
        # Get graph representation
        atom_fea, nbr_idx, nbr_fea, _ = self.cif_data[idx]
        
        # Get target properties from benchmark data
        mat_id = self.cif_paths[idx].split('/')[-1].split('.')[0].split('__')[0]
        targets = self.benchmark_data_map.get(mat_id, {})
        
        # Define the keys for our specific tasks
        task_keys = ['practical_capacity_mAh_g', 'voltage_vs_Li_Li_mV', 'cycle_life_cycles']
        target_values = [targets['properties'].get(k, 0.0) for k in task_keys]
        
        return (
            torch.tensor(atom_fea, dtype=torch.float),
            torch.tensor(nbr_idx, dtype=torch.long),
            torch.tensor(nbr_fea, dtype=torch.float),
            torch.tensor(target_values, dtype=torch.float)
        )

def get_data_loaders(config):
    """Main function to create train and validation dataloaders."""
    benchmark_data = BenchmarkSetLoader.load(config['paths']['benchmark_json'])
    
    # Create a map for easy lookup
    benchmark_map = {item['material_id']: item for item in benchmark_data}

    # Assume CIF paths are generated from the directory
    cif_paths = [f"./cif_files/{mat['material_id']}__dummy.cif" for mat in benchmark_data] # Placeholder paths
    
    dataset = GraphDataset(cif_paths, benchmark_map)
    
    val_split = config['training']['validation_split']
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader