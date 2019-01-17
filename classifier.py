import torch
from torch import nn
import torch.nn.functional as F

class ModelClassifier(nn.Module):
    """
    This class creates new classifier to update the pre-trained Neural Network.
    """
    def __init__(self, in_features, hidden_features, hidden_features2, out_features=102, drop_prob=0.25):
        """
        Function to create the classifier architecture with arbitrary hidden layers.
        Parameters:
         in_features: integer, pre-defined input for the network.
         hidden_features: integer, arbitrary hidden units decided by the user.
         hidden_features2: integer, pre-defined hidden units. 
         out_features: integer, 102 classified output.
         drop_prob: float, dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features2)
        self.fc3 = nn.Linear(hidden_features2, out_features)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, x):
        """ 
        Function to forward pass through the network.
        Parameters:
         x: tensor to pass through the network.
        Returns:
         x: output logits.
        """
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)       
        x = F.log_softmax(x, dim=1)        
        return x