'''
ARDMR code

Yibo Hu
Shanghai Jiao Tong University
2025.2
'''
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec




class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)


class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=16):
        super(Encoder, self).__init__()

        c = first_out_channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, kernel_size=3, stride=2, padding=1),  # 80
            ResBlock(2 * c),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1),  # 40
            ResBlock(4 * c),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4 * c, 8 * c, kernel_size=3, stride=2, padding=1),  # 20
            ResBlock(8 * c),
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8

        return [out0, out1, out2, out3]


class RegHead(nn.Module):
    def __init__(self, in_channels, channels=0):
        super(RegHead, self).__init__()
        self.conv3 = nn.Conv3d(in_channels, 3, 3, 1, 1)
        self.conv3.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv3.weight.shape))
        self.conv3.bias = nn.Parameter(torch.zeros(self.conv3.bias.shape))

    def forward(self, x):
        x = self.conv3(x)
        return x

class DefConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()
        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.SiLU()#nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class DMR(nn.Module):
    def __init__(self, inchannel, statechannel=8, vol_shape=[64, 160, 192]):
        super(DMR, self).__init__()
        self.encoder_moving = Encoder(in_channel=inchannel, first_out_channel=statechannel)
        self.encoder_fixed = Encoder(in_channel=inchannel, first_out_channel=statechannel)
        # self.encoder_fixed = Encoder(in_channel=inchannel, first_out_channel=statechannel)
        c = statechannel
        self.reghead4 = RegHead(in_channels=8*c)
        self.reghead3 = RegHead(in_channels=4*c)
        self.reghead2 = RegHead(in_channels=2*c)
        self.reghead1 = RegHead(in_channels=c)

        self.cconv_4 =  nn.Sequential(
            DefConvInsBlock(8*2 * c , 8*2 * c, 3, 1),
            DefConvInsBlock(8*2 * c, 8* c, 3, 1)
        )
        self.deconv_4= nn.Sequential(
            DefConvInsBlock(8*3 * c , 8* c, 3, 1),
        )
        self.upconv3 = UpConvBlock(8 * c, 4 * c, 4, 2)

        self.cconv_3 = nn.Sequential(
            DefConvInsBlock(4*3* c, 4*2 * c, 3, 1),
            DefConvInsBlock(4*2 * c, 4* c, 3, 1)
        )
        self.upconv2 = UpConvBlock(4 * c, 2 * c, 4, 2)
        self.cconv_2 = nn.Sequential(
            DefConvInsBlock(2*3* c, 2*2* c, 3, 1),
            DefConvInsBlock(2*2 * c, 2*c, 3, 1)
        )
        self.upconv1 = UpConvBlock(2 * c,   c, 4, 2)
        self.cconv_1 = nn.Sequential(
            DefConvInsBlock(1*3 * c, 1 *2* c, 3, 1),
            DefConvInsBlock(1*2 * c, c, 3, 1)
        )
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(4):
            self.warp.append(SpatialTransformer([s // 2**i for s in vol_shape]))
            self.diff.append(VecInt([s // 2 ** i for s in vol_shape]))

    def forward(self,moving,fixed):
        M1, M2, M3, M4 = self.encoder_moving(moving)
        F1, F2, F3, F4 = self.encoder_fixed(fixed)
        x_feas = [M1, M2, M3, M4]
        y_feas = [F1, F2, F3, F4]

        #----------金字塔配准---------------------
        C4 = torch.cat([F4, M4], dim=1)
        C4 = self.cconv_4(C4)
        flow_all = self.reghead4(C4)
        flow_all = self.diff[3](flow_all)

        # -------------------------------------

        wm4 = self.warp[3](M4, flow_all)
        C4 = torch.cat([F4,wm4,C4], dim=1)
        C4 = self.deconv_4(C4)
        flow = self.reghead4(C4)
        flow = self.diff[3](flow)

        flow_all = self.upsample_trilin(2* (self.warp[3](flow_all,flow)+flow))


        D3 = self.upconv3(C4)
        wm3 = self.warp[2](M3, flow_all)
        C3 = torch.cat([F3, wm3,D3], dim=1)
        C3 = self.cconv_3(C3)
        flow = self.reghead3(C3)
        flow = self.diff[2](flow)

        flow_all = self.upsample_trilin(2 * (self.warp[2](flow_all, flow) + flow))


        D2 = self.upconv2(C3)
        wm2 = self.warp[1](M2, flow_all)
        C2 = torch.cat([F2, wm2,D2], dim=1)
        C2 = self.cconv_2(C2)
        flow = self.reghead2(C2)
        flow = self.diff[1](flow)

        flow_all = self.upsample_trilin(2 * (self.warp[1](flow_all, flow) + flow))


        D1 = self.upconv1(C2)
        wm1 = self.warp[0](M1, flow_all)
        C1 = torch.cat([F1, wm1, D1], dim=1)
        C1 = self.cconv_1(C1)
        flow = self.reghead1(C1)
        flow = self.diff[0](flow)
        flow_all = self.warp[0](flow_all, flow) + flow

        warped_x = self.warp[0](moving, flow_all)

        return warped_x, x_feas, y_feas, flow_all


