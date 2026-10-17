import torch
import pickle
import pandas as pd
from models import ContextAwareSASRec

def run_inference_demo(model_path, mappings_path, metadata_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Mappings
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
    
    action2idx = mappings['action2idx']
    ref2idx = mappings['ref2idx']
    idx2ref = mappings['idx2ref']
    
    # Instantiate & Load Model
    model = ContextAwareSASRec(
        num_refs=mappings['NUM_REFS'],
        num_actions=mappings['NUM_ACTION_TYPES'],
        max_seq_len=20,
        hidden_dim=128,
        num_heads=4,
        dropout_rate=0.3
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Mock sequence
    test_actions = [
        action2idx.get('search for item', 1),
        action2idx.get('filter selection', 3),
        action2idx.get('clickout item', 1),
        action2idx.get('clickout item', 1),
        action2idx.get('change of sort order', 7),
    ]
    
    # Standard padded timeline setup
    test_refs = [0, 0, int(ref2idx.get('4', 1)), int(ref2idx.get('444', 2)), 0]
    pad = 20 - len(test_actions)
    actions_padded = [0] * pad + test_actions
    refs_padded    = [0] * pad + test_refs
    
    action_tensor = torch.tensor([actions_padded], dtype=torch.long).to(device)
    ref_tensor    = torch.tensor([refs_padded],    dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(ref_tensor, action_tensor)
        top10  = torch.topk(logits, 10, dim=1).indices[0].tolist()
        
    # Show Results
    print("Top 10 Predictions:")
    for rank, ref_id in enumerate(top10, 1):
        original_id = idx2ref.get(ref_id, "unknown")
        print(f"  {rank}. Hotel ID: {original_id}")

if __name__ == '__main__':
    print("Inference pipeline script loaded successfully.")