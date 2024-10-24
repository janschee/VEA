import torch
from torchvision import models

def alex_net(num_classes: int): 
    return models.AlexNet(num_classes=num_classes)

class AlexNet(torch.nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        # Conv layers
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            
            torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # AdaptiveAvgPool2d() allows flexible input sizes the output size is needed for the linear layers next
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6,6))
        
        # Linear layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten from the index=1 dim to the end
        logits = self.classifier(x)
        return logits