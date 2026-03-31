"""
Prediction script with Monte Carlo Dropout UQ interface.
"""
import torch
from models.cgcnn_mtl.py import CGCNN_MTL
import yaml

def load_model_and_predict(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = CGCNN_MTL(
        orig_atom_fea_len=92, # Assuming 92 unique elements
        nbr_fea_len=config['graph']['nbr_fea_len'],
        atom_fea_len=config['model']['atom_fea_len'],
        n_graph_conv=config['model']['n_graph_conv'],
        h_fea_len=config['model']['h_fea_len'],
        n_tasks=config['model']['n_tasks']
    )
    
    # Load pre-trained weights (in this case, randomly initialized)
    # model.load_state_dict(torch.load(checkpoint_path))
    # For this demo, we use the randomly initialized model
    model.eval()
    
    # Example prediction function
    def predict_with_uq(sample_data):
        """
        Predicts properties with uncertainty quantification using MC Dropout.
        """
        model.train() # Enable dropout for MC sampling
        all_predictions = []
        
        for _ in range(config['uq']['mc_dropout_iterations']):
            with torch.no_grad():
                # Note: sample_data should be a single batch from your dataloader
                atom_fea, nbr_idx, nbr_fea, _ = [d.unsqueeze(0) for d in sample_data] # Add batch dimension
                pred_tasks = model(atom_fea, nbr_idx, nbr_fea, None)
                # Concatenate predictions from all tasks for this iteration
                all_predictions.append(torch.cat(pred_tasks, dim=1)) 
        
        all_preds_tensor = torch.stack(all_predictions) # [n_iter, 1, n_tasks]
        
        mean_pred = torch.mean(all_preds_tensor, dim=0).squeeze(0) # [n_tasks]
        std_pred = torch.std(all_preds_tensor, dim=0).squeeze(0)   # [n_tasks]
        
        return mean_pred, std_pred

    return predict_with_uq

if __name__ == "__main__":
    predictor = load_model_and_predict('config.yaml', 'checkpoints/model.pth')
    print("Model loaded. Prediction function ready.")
    # Actual prediction would require a properly formatted sample from the data loader.
    # e.g., sample = next(iter(data_loader))
    # mean, std = predictor(sample)