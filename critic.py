import torch, tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from util import SpeechEnhancementData
from variables import *

def conv_block(in_channels, out_channels, final_layer = False):
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=2
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                        )
                    )  
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=1, 
                padding=2
                ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                        )
                    )    

def linear_block():
    return nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 5 * 4, 512),
                nn.Linear(512, 2),
                nn.Dropout(p=0.5),
                nn.Softmax(dim=1),
                        )          

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)
        self.conv4 = conv_block(64, 128, True)
        self.linear = linear_block()

        self.model = nn.Sequential(
                            self.conv1,
                            self.conv2,
                            self.conv3,
                            self.conv4,
                            self.linear
                                )   

    def forward(self, audio_signal):
        return self.model(audio_signal)
        
class CRITIC(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        cnn = CNN()
        self.cnn = cnn.to(self.device)
        self.data = SpeechEnhancementData()

        input_shape = self.data.__getitem__(0)[0][0].shape
        summary(self.cnn, input_shape)


        self.optimizer = optim.Adam(
                                self.cnn.parameters(), 
                                lr=learning_rate
                                    )
        self.loss_fn = nn.CrossEntropyLoss()


    def train_loop(self):
        n_batches = len(self.data) // batch_size
        for i_batches in tqdm.tqdm(range(n_batches)):
            X, Y = self.data.__getitem__(i_batches)
            self.optimizer.zero_grad()
            output = self.cnn(X)
            loss = self.loss_fn(output, Y)
            loss.backward()
            self.optimizer.step()
        print(loss.detach().cpu().item())

    def train(self):
        for epoch in range(epochs):
            self.train_loop()
        # torch.save(self.cnn.state_dict(), 'model.pt')

model = CRITIC()
model.train()
