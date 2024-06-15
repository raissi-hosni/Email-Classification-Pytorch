import torch.nn as nn
import torch.nn.functional as f

class NeuralNet(nn.Module):
    def __init__(self, inp_features,hidden_size1 = 100, hidden_size2 = 50, out_features = 2):
        super().__init__()#instantiate our nn.Module
        self.fc1 = nn.Linear(inp_features, hidden_size1)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)# Second fully connected layer
        self.out = nn.Linear(hidden_size2, out_features)# Output layer (2 classes: spam or ham)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)
        return x