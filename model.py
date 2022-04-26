import torch
import torch.nn as nn
from einops import rearrange

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ConvAttentionNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.embedding1 = nn.Linear(128, 1024)
        self.atten1 = Attention(1024, 8)
        self.embedding2 = nn.Linear(1024, 128)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.gelu = nn.GELU()
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.embedding3 = nn.Linear(512, 1024)
        self.atten2 = Attention(1024, 8)
        self.embedding4 = nn.Linear(1024, 512)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, 256),
                                        nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        xa = out.permute(0, 2, 3, 1)
        Xshape = xa.shape
        xa = xa.reshape(Xshape[0], Xshape[1] * Xshape[2], Xshape[3])
        xa = self.embedding1(xa)
        xa = xa + self.gelu(self.atten1(xa))
        xa = self.embedding2(xa)
        
        out = xa.reshape(Xshape[0], Xshape[1], Xshape[2], Xshape[3]).permute(0, 3, 1, 2)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        
        xa = out.permute(0, 2, 3, 1)
        Xshape = xa.shape
        xa = xa.reshape(Xshape[0], Xshape[1] * Xshape[2], Xshape[3])
        xa = self.embedding3(xa)
        xa = xa + self.gelu(self.atten2(xa))
        xa = self.embedding4(xa)
        
        out = xa.reshape(Xshape[0], Xshape[1], Xshape[2], Xshape[3]).permute(0, 3, 1, 2)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    X = torch.randn(2, 3, 32, 32)
    net = ConvAttentionNet(3, 10)
    net(X)