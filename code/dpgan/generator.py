import torch.nn as nn
from . import norms as norms
import torch
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
import math

def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out



def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin

class AdaGN(nn.Module):
    '''
    adaptive group normalization
    '''
    def __init__(self, n_channel, groups, style_dim):
        """
        ndim: dim of the input features 
        n_channel: number of channels of the inputs 
        ndim_style: channel of the style features 
        """
        super().__init__()
        self.n_channel = n_channel
        self.style_dim = style_dim
        self.groups = groups
        self.out_dim = n_channel * 2
        self.norm = nn.GroupNorm(self.groups, n_channel)
        in_channel = n_channel 
        self.emd = nn.Conv2d(self.style_dim, self.out_dim, 1, stride=1, padding=0)
        self.emd.bias.data[:in_channel] = 1
        self.emd.bias.data[in_channel:] = 0

    def __repr__(self):
        return f"AdaGN(GN(8, {self.n_channel}), Linear({self.style_dim}, {self.out_dim}))" 
        
    def forward(self, image, style):
        style = self.emd(style)
        factor, bias = style.chunk(2, 1)
        #print(image.size())
        result = self.norm(image)
        result = result * factor + bias  
        return result 

class DP_GAN_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        self.netFeat = opt.feat_extr
        
        if not self.opt.no_3dnoise:
            self.adaGN = AdaGN(self.opt.semantic_nc + self.opt.z_dim, 9, 32)
        else:
            self.adaGN = AdaGN(self.opt.semantic_nc, 4, 32)

        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))

        
        self.seg_pyrmid = nn.ModuleList([])
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(self.opt.semantic_nc, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)))

        self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        for i in range(len(self.channels)-2):
            self.seg_pyrmid.append(nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.tanh = nn.Tanh()

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None):
        seg = input
        # Extract features with the FPN of VolRecon
        feats = self.netFeat(seg[:, :3, :, :])
        
        # Add noise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        if not self.opt.no_3dnoise:
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=device)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        
        
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        feats = F.interpolate(feats, size=(16, 16), mode='bilinear')
        x = self.adaGN(x, feats)


        pyrmid = []
        s = seg
        for op in self.seg_pyrmid:
            s = op(s)
            pyrmid.append(s)

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            seg_f = torch.cat([F.interpolate(pyrmid[-1], size=(x.shape[2], x.shape[3]), mode='bilinear'), pyrmid[self.opt.num_res_blocks - i]], 1)
            x = self.body[i](x, seg_f)
            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = self.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out