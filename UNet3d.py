import torch.nn as nn
import torch
import torch.nn.functional as F



#Our implementation of the UNet3D with less channels for memory reasons
#The use of conv based residual improved segmentation significantly (ResUNet 3D)
class UNet3D(nn.Module):

    def __init__(self, num_input_channels=1,num_output_channels=4, feat_channels=[32, 64, 128, 256, 512]):
        
        super(UNet3D, self).__init__()
        
        # Encoder downsamplers
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.pool2 = nn.MaxPool3d((1,2,2))
        self.pool3 = nn.MaxPool3d((1,2,2))
        self.pool4 = nn.MaxPool3d((1,2,2))
        
        # Encoder convolutions
        self.conv_1 = Conv3D_Block(num_input_channels, feat_channels[0])
        self.conv_2 = Conv3D_Block(feat_channels[0], feat_channels[1])
        self.conv_3 = Conv3D_Block(feat_channels[1], feat_channels[2])
        self.conv_4 = Conv3D_Block(feat_channels[2], feat_channels[3])
        self.conv_5 = Conv3D_Block(feat_channels[3], feat_channels[4])

        # Decoder convolutions
        self.dec_conv_4 = Conv3D_Block(2*feat_channels[3], feat_channels[3])
        self.dec_conv_3 = Conv3D_Block(2*feat_channels[2], feat_channels[2])
        self.dec_conv_2 = Conv3D_Block(2*feat_channels[1], feat_channels[1])
        self.dec_conv_1 = Conv3D_Block(2*feat_channels[0], feat_channels[0])

        # Decoder upsamplers
        self.deconv_4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.final = nn.Conv3d(feat_channels[0], num_output_channels, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        
        # Encoder part
        
        x1 = self.conv_1(x)
        
        x_low1 = self.pool1(x1)
        x2 = self.conv_2(x_low1)
        
        x_low2 = self.pool2(x2)
        x3 = self.conv_3(x_low2)
        
        x_low3 = self.pool3(x3)
        x4 = self.conv_4(x_low3)
        
        x_low4 = self.pool4(x4)
        base = self.conv_5(x_low4)

        # Decoder part
        
        d4 = torch.cat([self.deconv_4(base), x4], dim=1)
        d_high4 = self.dec_conv_4(d4)

        d3 = torch.cat([self.deconv_3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_3(d3)

        d2 = torch.cat([self.deconv_2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_2(d2)

        d1 = torch.cat([self.deconv_1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_1(d1)
        
        #final layer
        seg = self.final(d_high1)

        return seg

#The inner convolutions within each level of the UNet
class Conv3D_Block(nn.Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1):
        
        super(Conv3D_Block, self).__init__()

        self.conv1 = nn.Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.batch_norm1 = nn.BatchNorm3d(out_feat)
        self.conv2 = nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.batch_norm2 = nn.BatchNorm3d(out_feat)
        #Use to compute residual convolution
        self.conv3 = nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        #keep input as a residual
        res = x
        x = self.batch_norm1(self.conv1(x))
        x = F.relu(x)
        x = self.batch_norm2(self.conv2(x))
        x = F.relu(x)

        #sum output and residual
        return x + self.conv3(res)

#Upsampler block to reconstruct image
class Deconv3D_Block(nn.Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv =  nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel), 
                                       stride=(1,stride,stride), padding=(0, padding, padding), 
                                       output_padding=0, bias=True)
    
    def forward(self, x):
        return F.relu(self.deconv(x))