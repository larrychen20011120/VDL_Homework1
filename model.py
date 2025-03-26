import torch
import torch.nn as nn
from torchvision import models

class HundredClassResNet(nn.Module):
    def __init__(self):
        super(HundredClassResNet, self).__init__()
        
        self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        # self.resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)

        for param in self.resnet.parameters():
            param.requires_grad = True
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 100),
        )

        torch.nn.init.kaiming_normal_(self.resnet.fc[1].weight) # Initialize weights with Kaiming Normal
        torch.nn.init.zeros_(self.resnet.fc[1].bias) # Initialize bias as 0

    def forward(self, x):
        return self.resnet(x)

    
if __name__ == "__main__":
    model = HundredClassResNet()
    print(f"The # of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    model.eval()
    print(model(torch.randn(1, 3, 300, 300)).size())