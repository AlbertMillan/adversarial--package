import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os

from .modules import activation


class BasicBlock(nn.Module):
    """ Assumes kernel size of constant size."""
    
    def __init__(self, activation_cfg, in_channel, out_channel, stride, padding):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=stride, padding=padding)
        self.batchNorm1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.9)
        self.activation = activation(activation_cfg)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.batchNorm1(out)
        out = self.activation(out)
        
        return out

    
    
class NetworkBlock(nn.Module):
    """ Creates a CX block. """
    
    def __init__(self, n_blocks, block, activation_cfg, in_planes, out_planes, padding):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(n_blocks, block, activation_cfg, in_planes, out_planes, padding)
        
    def _make_layer(self, n_blocks, block, activation_cfg, in_planes, out_planes, padding):
        """ Creates a CX block as nn.Sequential() """
        layers = []
        is_three_block = (n_blocks == 3)
        for i in range(int(n_blocks)):
            # Stride is 2x2 for first convolution of C3 blocks.
            stride = (2 if i == 0 and is_three_block else 1 )
            layers.append( block(activation_cfg, in_planes[i], out_planes[i], stride, padding) )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class Denoiser(nn.Module):
    
    def __init__(self, denoiser_cfg, kernel_size=(3,3), stride=1, padding=1):
        super(Denoiser, self).__init__()
        
        # These can probably be stored in a configuration file of some sort.
        nChannels_fwd = np.array([3, 64, 128, 256, 256, 256])
        blockSzs_fwd = np.array([2, 3, 3, 3, 3])
        nChannels_bwd = np.array([256, 256, 256, 128, 64, 3])
        blockSzs_bwd = np.array([3, 3, 3, 2])
        
        # Define (H, W) for upsampling
        h = []
        w = []
        x_h = denoiser_cfg.INPUT_HEIGHT
        x_w = denoiser_cfg.INPUT_WIDTH
        for i in range(len(nChannels_bwd) - 1):
            h.append(x_h)
            w.append(x_w)
            x_h = int(np.ceil(x_h / 2.))
            x_w = int(np.ceil(x_w / 2.))

        # Encoder
        self.forward_blocks = self._make_CX_fwd_blocks(blockSzs_fwd, nChannels_fwd, denoiser_cfg.ACTIVATION)
        
        # Decoder
        self.backward_blocks = self._make_CX_bwd_blocks(blockSzs_bwd, nChannels_bwd, nChannels_fwd, denoiser_cfg.ACTIVATION)
        
        # Upsampling on decoder fusion layers
        self.upsampler = self._make_upsample_layers(blockSzs_bwd, nChannels_bwd, h, w)
        
        # 1v1 convolution
        self.conv_1 = nn.Conv2d(nChannels_bwd[-2], nChannels_bwd[-1], kernel_size=(1,1), stride=stride, padding=0)
        
        
        
    def _make_CX_fwd_blocks(self, blockSzs, nChannels, activation_cfg):
        """ 
        Creates a the list of network blocks CX for encoder as nn.ModuleList().
        We want to be able to retrieve outputs of each block, hence
        store as nn.ModuleList() as opposed nn.Sequential().
        """
        block = BasicBlock
        blocks = []
        for i in range(len(blockSzs)):
            in_planes = np.append(nChannels[i],  [nChannels[i+1]] * (blockSzs[i] - 1 ) )
            out_planes = [nChannels[i+1]] * blockSzs[i]
            blocks.append( NetworkBlock(blockSzs[i], block, activation_cfg, in_planes, out_planes, padding=1) )
                         
        return nn.ModuleList(blocks)
    
    def _make_CX_bwd_blocks(self, blockSzs, nChannels_bwd, nChannels_fwd, activation_cfg):
        """ 
        Creates a the list of network blocks CX for decoder as nn.ModuleList().
        """
        last_fusion_dim = len(nChannels_fwd) - 2
        
        block = BasicBlock
        blocks = []
        for i in range(len(blockSzs)):
            in_planes = np.append(nChannels_bwd[i] + nChannels_fwd[last_fusion_dim-i],  
                                  [nChannels_bwd[i+1]] * (blockSzs[i] - 1 ) )
            out_planes = [nChannels_bwd[i+1]] * blockSzs[i]
            blocks.append( NetworkBlock(blockSzs[i], block, activation_cfg, in_planes, out_planes, padding=1) )
                         
        return nn.ModuleList(blocks)
    
    def _make_upsample_layers(self, blockSzs, nChannels, h, w):
        """ Returns list with upsampling operation at each fusion layer, top-to-bottom. """
        upsampler = []
        for i in range(len(blockSzs)):
            upsampler.insert(0, nn.Upsample(size=(h[i], w[i]), mode='bilinear', align_corners=False) )
        
        return nn.ModuleList(upsampler)
        
    
    def forward(self, out):
        
        # Each output has its own shape
        layer_outputs = []
        
        
        # ModuleList can act as an iterable
        for i in range(len(self.forward_blocks)):
            out = self.forward_blocks[i](out)

            # Store block outputs
            if i != (len(self.forward_blocks) - 1):
                layer_outputs.insert(0, out)
                
        
        for i in range(len(self.backward_blocks)):
            # 1. Fusion layer
            out = self.upsampler[i](out)
            
            # 2. Append retained output
            out = torch.cat((out, layer_outputs[i]), dim=1)
            
            # 3. Perform backward convolution block
            out = self.backward_blocks[i](out)
            
        out = self.conv_1(out)
        
        return out
        
        
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    x = torch.randn(10, 3, 32, 32, device='cuda:0')
    denoiser = Denoiser(x.size(2), x.size(3)).cuda()
    denoiser.forward(x)
    
        