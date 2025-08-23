import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, lstm_layers=1):
        super(CNNLSTM, self).__init__()

        base_cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(base_cnn.children())[:-1])

        self.feature_dim = 512
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)            
        features = self.cnn(x).squeeze()      

        features = features.view(B, T, -1)   
        lstm_out, _ = self.lstm(features)   
        last_output = lstm_out[:, -1, :]   

        out = self.classifier(last_output)
        return out