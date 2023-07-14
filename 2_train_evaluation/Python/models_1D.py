import torch
import torch.nn as nn
from functools import partial

class cnn(nn.Module):
    def __init__(self, num_input_channels, segment_length, device ) -> None:
        super().__init__()
        self.device = device
        self.block_sizes = [num_input_channels*2**i for i in range(0,6)]
        self.downsampling = 2
        self.cnn_output_size = int(segment_length / self.downsampling**(len(self.block_sizes)-1)) # make sure segments dividable by 2^x times x = num of downsampling
        # Create 1D residual cnn
        self.convNet = ResNetEncoder(downsampling=self.downsampling, blocks_sizes = self.block_sizes, deepths=[1]*(len(self.block_sizes)-1))
        # Last layer - Fully connected layer
        self.fcn = nn.Sequential(
            nn.Linear(self.cnn_output_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.convNet(x)
        x = self.fcn(x.view(x.shape[0], x.shape[1]*x.shape[2])) 
        return  x

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, )# dynamic add padding based on the kernel_size because can't just use 'same' if stride !=1

conv121x1 = partial(Conv1dAuto, kernel_size = 121, bias=False) #kernel size must be odd
conv3x1 = partial(Conv1dAuto, kernel_size = 3, bias=False)
# conv = conv25x1(in_channels=32, out_channels=64)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv121x1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            # activation_func(self.activation),
            torch.nn.Dropout(p=0.2),
            # conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=True),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1,downsampling=2, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = downsampling if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, downsampling=2, blocks_sizes=[1, 16, 32, 64, 128], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, conv=conv3x1, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, downsampling=downsampling, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths)],
            nn.Conv1d(blocks_sizes[-1],1,kernel_size=1, padding = 0),
            nn.BatchNorm1d(1),
        ])
       
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
   
# helper functions
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    # Instantiate a convolutional layer as defined by conv e.g conv1d, or conv25x1
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm1d(out_channels))

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)],
        ['gelu', nn.GELU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]
    
    