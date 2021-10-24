import torch
import torch.nn as nn
from torchsummary import summary

from variables import *

def conv_block(in_channels, out_channels, kernel_size=32, stride=2, padding=15):
    return nn.Conv1d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding
                )

def upsample_block(in_channels, out_channels, kernel_size=32, stride=2, padding=15):
    return nn.ConvTranspose1d(
                    in_channels = in_channels, 
                    out_channels = out_channels, 
                    kernel_size = kernel_size, 
                    stride = stride, 
                    padding = padding
                    )

class Generator(nn.Module):
    def __init__(self, init_out_channels=16):
        super().__init__()

        '''
        Encoder inputs:
            x: noisy signal
            x.shape = (batch_size, 1, window_size)
           
        Encoder outputs / Decoder inputs:
            c: feature map 

        Decoder outputs:
            x_hat: denoised / enhanced signal
            -> each decoder output are concatenated with homologous encoder output, so the feature map sizes are doubled
        '''

        ############################ Encoder ############################
        self.encoder1 = conv_block(1, init_out_channels)
        self.encoder1_act = nn.PReLU()

        self.encoder2 = conv_block(init_out_channels, init_out_channels * 2)
        self.encoder2_act = nn.PReLU()

        self.encoder3 = conv_block(init_out_channels * 2, init_out_channels * 2)
        self.encoder3_act = nn.PReLU()

        self.encoder4 = conv_block(init_out_channels * 2, init_out_channels * 4)
        self.encoder4_act = nn.PReLU()

        self.encoder5 = conv_block(init_out_channels * 4, init_out_channels * 4)
        self.encoder5_act = nn.PReLU()

        self.encoder6 = conv_block(init_out_channels * 4, init_out_channels * 8)
        self.encoder6_act = nn.PReLU()

        self.encoder7 = conv_block(init_out_channels * 8, init_out_channels * 8)
        self.encoder7_act = nn.PReLU()

        self.encoder8 = conv_block(init_out_channels * 8, init_out_channels * 16)
        self.encoder8_act = nn.PReLU()

        self.encoder9 = conv_block(init_out_channels * 16, init_out_channels * 16)
        self.encoder9_act = nn.PReLU()
        
        self.encoder10 = conv_block(init_out_channels * 16, init_out_channels * 32)
        self.encoder10_act = nn.PReLU()

        self.encoder11 = conv_block(init_out_channels * 32, init_out_channels * 64)
        self.encoder11_act = nn.PReLU()

        ############################ Decoder ############################

        self.decoder10 = upsample_block(init_out_channels * 128, init_out_channels * 32)
        self.decoder10_act = nn.PReLU()

        self.decoder9 = upsample_block(init_out_channels * 64, init_out_channels * 16)
        self.decoder9_act = nn.PReLU()

        self.decoder8 = upsample_block(init_out_channels * 32, init_out_channels * 16)
        self.decoder8_act = nn.PReLU()

        self.decoder7 = upsample_block(init_out_channels * 32, init_out_channels * 8)
        self.decoder7_act = nn.PReLU()

        self.decoder6 = upsample_block(init_out_channels * 16, init_out_channels * 8)
        self.decoder6_act = nn.PReLU()

        self.decoder5 = upsample_block(init_out_channels * 16, init_out_channels * 4)
        self.decoder5_act = nn.PReLU()

        self.decoder4 = upsample_block(init_out_channels * 8, init_out_channels * 4)
        self.decoder4_act = nn.PReLU()

        self.decoder3 = upsample_block(init_out_channels * 8, init_out_channels * 2)
        self.decoder3_act = nn.PReLU()

        self.decoder2 = upsample_block(init_out_channels * 4, init_out_channels * 2)
        self.decoder2_act = nn.PReLU()

        self.decoder1 = upsample_block(init_out_channels * 4, init_out_channels)
        self.decoder1_act = nn.PReLU()

        self.decoder_out = upsample_block(init_out_channels * 2, 1)
        self.decoder_out_act = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x, z):
        """
        Forward pass of generator.
        Args:
            x: input batch (signal)
            z: latent vector
        """

        enc1 = self.encoder1(x)
        enc1_act = self.encoder1_act(enc1)

        enc2 = self.encoder2(enc1_act)
        enc2_act = self.encoder2_act(enc2)

        enc3 = self.encoder3(enc2_act)
        enc3_act = self.encoder3_act(enc3)

        enc4 = self.encoder4(enc3_act)
        enc4_act = self.encoder4_act(enc4)

        enc5 = self.encoder5(enc4_act)
        enc5_act = self.encoder5_act(enc5)

        enc6 = self.encoder6(enc5_act)
        enc6_act = self.encoder6_act(enc6)

        enc7 = self.encoder7(enc6_act)
        enc7_act = self.encoder7_act(enc7)

        enc8 = self.encoder8(enc7_act)
        enc8_act = self.encoder8_act(enc8)

        enc9 = self.encoder9(enc8_act)
        enc9_act = self.encoder9_act(enc9)

        enc10 = self.encoder10(enc9_act)
        enc10_act = self.encoder10_act(enc10)

        enc11 = self.encoder11(enc10_act)
        enc11_act = self.encoder11_act(enc11)
        
        encoder_out = torch.cat((enc11_act, z), dim=1)

        dec10 = self.decoder10(encoder_out)

        dec10_cat = torch.cat((dec10, enc10), dim=1)
        dec10_act = self.decoder10_act(dec10_cat)

        dec9 = self.decoder9(dec10_act)
        dec9_cat = torch.cat((dec9, enc9), dim=1)
        dec9_act = self.decoder9_act(dec9_cat)

        dec8 = self.decoder8(dec9_act)
        dec8_cat = torch.cat((dec8, enc8), dim=1)
        dec8_act = self.decoder8_act(dec8_cat)

        dec7 = self.decoder7(dec8_act)
        dec7_cat = torch.cat((dec7, enc7), dim=1)
        dec7_act = self.decoder7_act(dec7_cat)

        dec6 = self.decoder6(dec7_act)
        dec6_cat = torch.cat((dec6, enc6), dim=1)
        dec6_act = self.decoder6_act(dec6_cat)

        dec5 = self.decoder5(dec6_act)
        dec5_cat = torch.cat((dec5, enc5), dim=1)
        dec5_act = self.decoder5_act(dec5_cat)

        dec4 = self.decoder4(dec5_act)
        dec4_cat = torch.cat((dec4, enc4), dim=1)
        dec4_act = self.decoder4_act(dec4_cat)

        dec3 = self.decoder3(dec4_act)
        dec3_cat = torch.cat((dec3, enc3), dim=1)
        dec3_act = self.decoder3_act(dec3_cat)

        dec2 = self.decoder2(dec3_act)
        dec2_cat = torch.cat((dec2, enc2), dim=1)
        dec2_act = self.decoder2_act(dec2_cat)

        dec1 = self.decoder1(dec2_act)
        dec1_cat = torch.cat((dec1, enc1), dim=1)
        dec1_act = self.decoder1_act(dec1_cat)

        dec_final = self.decoder_out(dec1_act)
        output = self.decoder_out_act(dec_final)
        return output

cnn = Generator()
input_shape = [(1 , 16384), (1024 , 8)]
summary(cnn, input_shape)