import torch.nn as nn
import torchvision.transforms as T


class CNN002(nn.Module):
    def __init__(self, freqConvParam, temporalConvParam, inputSize):
        super().__init__()
        self.frequencyConvolution = nn.Sequential(
            nn.Conv2d(*freqConvParam),
            nn.BatchNorm2d(freqConvParam[1]),
            nn.ReLU()
        )
        self.temporalConvolution = nn.Sequential(
            nn.Conv2d(*temporalConvParam),
            nn.BatchNorm2d(temporalConvParam[1]),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # print(inputSize)
        inputSize = (
            freqConvParam[1],
            (inputSize[1]-freqConvParam[2][0] + freqConvParam[4]+ freqConvParam[3])// freqConvParam[3],
            (inputSize[2]-freqConvParam[2][1] + freqConvParam[4]+ freqConvParam[3])// freqConvParam[3]
        )
        # print(inputSize)
        inputSize = (
            temporalConvParam[1],
            (inputSize[1]-temporalConvParam[2][0] + temporalConvParam[4]+ temporalConvParam[3])// temporalConvParam[3],
            (inputSize[2]-temporalConvParam[2][1] + temporalConvParam[4]+ temporalConvParam[3])// temporalConvParam[3]
        )
        # print(inputSize)
        self.fcl = nn.Sequential(
            nn.Linear(inputSize[0] * inputSize[1] * inputSize[2],512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, x):
        out = self.frequencyConvolution(x)
        out = self.temporalConvolution(out)
        out = self.flatten(out)
        logits = self.fcl(out)
        return logits


class VGG_Style_3Block(nn.Module):
    def __init__(self, H=224, W=224, in_channels=3):
        super().__init__()
        self.W = W
        self.H = H
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(128*W*H//64, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, X):
        X_resized = T.Resize((self.H, self.W), antialias=None)(X)
        return self.net(X_resized)
