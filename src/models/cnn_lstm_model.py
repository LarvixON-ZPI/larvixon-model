import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, lstm_layers=1, temporal_pooling='last'):
        super(CNNLSTM, self).__init__()

        base_cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_cnn.children())[:-1])

        self.feature_dim = 512
        self.temporal_pooling = temporal_pooling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        if temporal_pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x).squeeze()

        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        
        if self.temporal_pooling == 'mean':
            pooled_output = torch.mean(lstm_out, dim=1)
        elif self.temporal_pooling == 'max':
            pooled_output, _ = torch.max(lstm_out, dim=1)
        elif self.temporal_pooling == 'attention':
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled_output = torch.sum(lstm_out * attention_weights, dim=1)
        elif self.temporal_pooling == 'adaptive':
            avg_pool = torch.mean(lstm_out, dim=1)
            max_pool, _ = torch.max(lstm_out, dim=1)
            pooled_output = (avg_pool + max_pool) / 2
        else:
            pooled_output = lstm_out[:, -1, :]

        out = self.classifier(pooled_output)
        return out
