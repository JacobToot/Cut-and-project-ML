import torch
state_dict = torch.load("/Users/jacobtoot/Documents/Documents - Jacob’s MacBook Pro/GitHub/cut-and-project-ML/results/npe_1d/npe_1d_thesis/best_weights.pt", map_location="cpu")
print(list(state_dict.keys())[:10])