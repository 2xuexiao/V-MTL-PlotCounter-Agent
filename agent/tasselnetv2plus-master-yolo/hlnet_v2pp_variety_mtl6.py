# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import warnings


class CustomLinearTransform(nn.Module):
    def __init__(self):
        super(CustomLinearTransform, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(x)
        return x * 2 + 1

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GlobalSeed(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = CustomLinearTransform()
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x)
        x = self.global_avg_pool(x)
        #print(x.shape)
        x = self.linear(x)
        # return x
        return x

class Conv_CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        # return x
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimConv(nn.Module):
    '''Normal Conv with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.SiLU()

        # self.nonlinearity = nn.ReLU()

        # if use_se:
        #     self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        # else:
        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0  # mg
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_dense(inputs))
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, isTrue=None):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels)  # mg
        self.block = nn.Sequential(*(RepVGGBlock(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class InjectionMultiSum_Auto_pool1(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            trans_channels: int
    ) -> None:
        super().__init__()
        trans_channels = [64, 32]
        self.trans_channels = trans_channels
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(trans_channels[0], oup, 1, act=False)
        self.global_act = Conv(trans_channels[0], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):

        x_l, x_g = x
        low_global_info = x_g.split(self.trans_channels, dim=1)

        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(low_global_info[0])
        global_feat = self.global_embedding(low_global_info[0])

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSum_Auto_pool2(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            trans_channels: int
    ) -> None:
        super().__init__()
        trans_channels = [64, 32]
        self.trans_channels = trans_channels
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(trans_channels[1], oup, 1, act=False)
        self.global_act = Conv(trans_channels[1], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_g = x
        low_global_info = x_g.split(self.trans_channels, dim=1)

        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(low_global_info[1])
        global_feat = self.global_embedding(low_global_info[1])

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSum_Auto_pool3(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            trans_channels: int
    ) -> None:
        super().__init__()
        trans_channels = [64, 128]
        self.trans_channels = trans_channels
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(trans_channels[0], oup, 1, act=False)
        self.global_act = Conv(trans_channels[0], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_g = x
        low_global_info = x_g.split(self.trans_channels, dim=1)

        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(low_global_info[0])
        global_feat = self.global_embedding(low_global_info[0])

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSum_Auto_pool4(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            trans_channels: int
    ) -> None:
        super().__init__()
        trans_channels = [64, 128]
        self.trans_channels = trans_channels

        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(trans_channels[1], oup, 1, act=False)
        self.global_act = Conv(trans_channels[1], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        x_l, x_g = x
        low_global_info = x_g.split(self.trans_channels, dim=1)

        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(low_global_info[1])
        global_feat = self.global_embedding(low_global_info[1])

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class LAF_px(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1_ = Conv(in_channel_list[0], out_channels, 1, 1)
        self.cv1 = Conv(in_channel_list[1], out_channels, 1, 1)
        self.cv_fuse = Conv(out_channels * 3, out_channels, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.downsample(x[0], output_size)
        x0 = self.cv1_(x0)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class low_FAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, y_l, x_m, y_m, x_s, y_s, y_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        y_n = F.interpolate(y_n, size=(H, W), mode='bilinear', align_corners=False)
        y_l = self.avg_pool(y_l, output_size)
        y_m = self.avg_pool(y_m, output_size)
        out = torch.cat([x_l, y_l, x_m, y_m, x_s, y_s, y_n], 1)
        return out


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.SiLU()

        # self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def fusevggforward(self, x):
        return self.nonlinearity(self.rbr_dense(x))


class low_IFM(nn.Module):
    def __init__(self, inc, embed_dim_p=128, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum([64, 64]))
        )

    def forward(self, x):
        return self.conv(x)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


class LAF_h(nn.Module):
    def forward(self, x):
        x1, x2 = x

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)


class TopBasicLayer(nn.Module):
    def __init__(self, input, embedding_dim, mlp_ratio=4., attn_ratio=2., drop=0., drop_path=0.):
        super().__init__()
        depths = 2
        key_dim = 8
        num_heads = 4
        trans_channels = [64, 32, 64, 128]
        self.block_num = depths

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            ))

        self.conv_1x1_n = nn.Conv2d(embedding_dim, sum(trans_channels[2:4]), 1)

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        x = self.conv_1x1_n(x)
        return x


class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)

        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Encoder(nn.Module):
    def __init__(self, arc='tasselnetv2plus'):
        super(Encoder, self).__init__()
        if arc == 'tasselnetv2':
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2), stride=2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
        elif arc == 'tasselnetv2plus':
            self.outy1 = nn.Sequential(Conv(3, 64, 6, 2, 2))
            self.outy2 = nn.Sequential(Conv(64, 128, 3, 2), C3(128, 128))
            self.outy3 = nn.Sequential(Conv(128, 256, 3, 2), C3(256, 256))
            self.outy4 = nn.Sequential(SPPF(256, 256, 5), Conv(256, 512, 3, 2), C3(512, 512))

            self.outx1 = nn.Sequential(Conv_CBAM(3, 16, 3, p=1),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d((2, 2), stride=2),
                                       )
            self.outx2 = nn.Sequential(Conv_CBAM(16, 32, 3, p=1),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d((2, 2), stride=2),
                                       )
            self.outx3 = nn.Sequential(Conv_CBAM(32, 64, 3, p=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.MaxPool2d((2, 2), stride=2),
                                       Conv_CBAM(64, 128, 3, p=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       Conv_CBAM(128, 128, 3, p=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(inplace=True)
                                       )
            self.outz11 = nn.Sequential(Conv_CBAM(80, 64, 3, p=1))
            self.outz21 = nn.Sequential(Conv_CBAM(160, 128, 3, p=1))
            self.outz31 = nn.Sequential(Conv_CBAM(384, 256, 3, p=1))
            self.outz12 = nn.Sequential(Conv_CBAM(80, 16, 3, p=1))
            self.outz22 = nn.Sequential(Conv_CBAM(160, 32, 3, p=1))
            self.outz32 = nn.Sequential(Conv_CBAM(384, 128, 3, p=1))
            self.low_fam = low_FAM()
            self.low_ifm = low_IFM(inc=1136)
            self.conv5 = Conv(128, 128, 3, 1)
            self.globalSeed = GlobalSeed(128, 16, 3, p=1)
            self.classifier = nn.Sequential(
                Conv(128, 64, 3, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(64, 2))
        else:
            raise NotImplementedError

    def forward(self, x):
        y1 = self.outy1(x)
        x1 = self.outx1(x)
        z1 = torch.cat([x1, y1], dim=1)
        z11 = self.outz11(z1)
        z12 = self.outz12(z1)
        y2 = self.outy2(z11)
        x2 = self.outx2(z12)
        z2 = torch.cat([x2, y2], dim=1)
        z21 = self.outz21(z2)
        z22 = self.outz22(z2)
        y3 = self.outy3(z21)
        x3 = self.outx3(z22)
        z3 = torch.cat([x3, y3], dim=1)
        z31 = self.outz31(z3)
        z32 = self.outz32(z3)
        y4 = self.outy4(z31)

        f1 = self.low_fam([z11, z12, z21, z22, z31, z32, y4])
        f2 = self.low_ifm(f1)
        encoderx = self.conv5(f2)
        classx = self.classifier(encoderx)
        return encoderx, classx


class Counter(nn.Module):
    def __init__(self, arc='tasselnetv2plus', input_size=64, output_stride=8):
        super(Counter, self).__init__()
        k = int(input_size / 8)
        avg_pool_stride = int(output_stride / 8)

        if arc == 'tasselnetv2':
            self.counter = nn.Sequential(
                nn.Conv2d(128, 128, (k, k), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)
            )
        elif arc == 'tasselnetv2plus':
            self.counter = nn.Sequential(
                nn.AvgPool2d((k, k), stride=avg_pool_stride),
                Conv_CBAM(128, 128, 1),
                nn.Conv2d(128, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1)
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.counter(x)
        return x


class Normalizer:
    @staticmethod
    def cpu_normalizer(x, imh, imw, insz, os):
        # CPU normalization
        bs = x.size()[0]
        normx = np.zeros((imh, imw))
        norm_vec = dense_sample2d(normx, insz, os).astype(np.float32)
        x = x.cpu().detach().numpy().reshape(bs, -1) * norm_vec
        return x

    @staticmethod
    def gpu_normalizer(x, imh, imw, insz, os):
        _, _, h, w = x.size()
        accm = torch.cuda.FloatTensor(1, insz * insz, h * w).fill_(1)
        accm = F.fold(accm, (imh, imw), kernel_size=insz, stride=os)
        accm = 1 / accm
        accm /= insz ** 2
        accm = F.unfold(accm, kernel_size=insz, stride=os).sum(1).view(1, 1, h, w)
        x *= accm
        return x.squeeze().cpu().detach().numpy()


class CountingModels(nn.Module):
    def __init__(self, arc='tasselnetv2plus', input_size=64, output_stride=8):
        super(CountingModels, self).__init__()
        self.input_size = input_size
        self.output_stride = output_stride

        self.encoder = Encoder(arc)
        self.counter = Counter(arc, input_size, output_stride)
        if arc == 'tasselnetv2':
            self.normalizer = Normalizer.cpu_normalizer
        elif arc == 'tasselnetv2plus':
            self.normalizer = Normalizer.gpu_normalizer

        self.weight_init()

    def forward(self, x, is_normalize=True):
        imh, imw = x.size()[2:]
        x, classx = self.encoder(x)
        x = self.counter(x)
        if is_normalize:
            x = self.normalizer(x, imh, imw, self.input_size, self.output_stride)
        return x, classx

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight,
                #         mode='fan_in',
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    from time import time

    insz, os = 64, 8
    imH, imW = 1080, 1920
    net = CountingModels(arc='tasselnetv2plus', input_size=insz, output_stride=os).cuda()
    with torch.no_grad():
        net.eval()
        x = torch.randn(1, 3, imH, imW).cuda()
        y = net(x)
        print(y.shape)

    import numpy as np

    with torch.no_grad():
        frame_rate = np.zeros((100, 1))

        for i in range(100):
            x = torch.randn(1, 3, imH, imW).cuda()
            torch.cuda.synchronize()
            start = time()

            y = net(x)

            torch.cuda.synchronize()
            end = time()

            running_frame_rate = 1 * float(1 / (end - start))
            frame_rate[i] = running_frame_rate
        print(np.mean(frame_rate))