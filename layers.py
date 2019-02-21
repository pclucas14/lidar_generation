import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import time

from utils import batch_pairwise_dist

# --------------------------------------------------------------------------
# Define rotation equivariant layers
# -------------------------------------------------------------------------- 

class round_conv2d(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=(0,0), bias=True):
        if isinstance(padding, int):
            padding = (padding, padding)

        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=(padding[0], 0), bias=bias)
        self.padding_ = padding[1]

    def forward(self, x):
        # first, we pad the input
        input = x
        if self.padding_ > 0:
            x = torch.cat([x[:, :, :, -self.padding_:], x, x[:, :, :, :self.padding_]], dim=-1)
        out = super().forward(x)
        return out

class round_deconv2d(nn.ConvTranspose2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=(0,0), bias=True):
        if isinstance(padding, int):
            padding = (padding, padding)
        
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=(padding[0], 0), bias=bias)
        
        self.padding_ = padding[1]

    def forward(self, x):
        input = x
        if self.padding_ > 0:
           x = torch.cat([x[:, :, :, -self.padding_:], x, x[:, :, :, 0:self.padding_]], dim=-1)
        out = super().forward(x)
        return out


# --------------------------------------------------------------------------
# Attention Convolution
# -------------------------------------------------------------------------- 

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation='relu', rel_dist_scaling=0):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.rel_dist_scaling = rel_dist_scaling
       

        out_c = max(in_dim//8, 1)

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_c , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_c , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    
    def forward(self,x, points=None):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width ,height = x.size() # (bs, C, H, W)
        if points is not None:
            # calculate distance
            factor = points.size(-1) // x.size(-1)
            points = points[:, :, ::factor, ::factor]
            offset = points.size(2) - x.size(2)
            points = points[:, :, offset:]
            points = points.reshape(points.size(0), points.size(1), -1)
            distance = batch_pairwise_dist(points, points)
            attention = self.softmax((distance.max() - distance) * 5)

        else:
            proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # (bs, H*W, c_out)
            proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # (bs, c_out, H*W)
        
            energy = torch.bmm(proj_query,proj_key) # transpose check

            attention = self.softmax(energy) # BX (N) X (N) 
       
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma * out + x
        return out # ,attention




class flex_conv_op(nn.Module):
    """ flex convolution backend """ 

    def __init__(self, ks=3, stride=1):
        super(flex_conv_op, self).__init__()
        self.ks = ks
        self.stride = stride

    def forward(self, x, kernel, points):
        # points : bs x 3 x h x w
        # kernel : c' x c x 4
        # x      : bs x c x h x w

        out_c, _, _ = kernel.size()
        ks = self.ks
        st = self.stride
        bs, in_c, h, w = x.size()
        
        # [52, 73, 31, 50, 2, 2]
        strided_x = x.as_strided((bs, 
                                  in_c, 
                                  (h - ks) // st + 1, 
                                  (w - ks) // st + 1, ks, ks), 
                                  (in_c * h * w, h * w, st * w, st, w, 1))
        
        in_c = 3
        # [52, 3, 31, 50, 2, 2]
        strided_points = points.as_strided((bs, 
                                  in_c, 
                                  (h - ks) // st + 1, 
                                  (w - ks) // st + 1, ks, ks), 
                                  (in_c * h * w, h * w, st * w, st, w, 1))
       
        ks = self.ks
        bd = ks // 2 if self.stride == 1 else ks // 4
            
        rel_pts = points[:, :, bd:-bd, bd:-bd][:, :, 0::st, 0::st].unsqueeze(-1)\
            .unsqueeze(-1) - strided_points

        # we need to add a row of 1's to capture the bias term
        # [52, 4, 31, 50, 2, 2]
        rel_pts = torch.cat([rel_pts, torch.ones_like(rel_pts[:, [0]])], dim=1)
        output = torch.einsum('bdhwkf,bihwkf,oid->bohw', (rel_pts, strided_x, kernel))
        return output


class flex_conv(nn.Module):
    """ flex convolution on a grid """

    def __init__(self, in_channels, out_channels, ks, stride=1, padding='same'):
        '''
        padding options are : 
        same  : zero padding such that (H',W') == (H,W)
        None  : no padding
        round : same padding, but with features from opposite sides
        '''
        super(flex_conv, self).__init__()
        weight = torch.FloatTensor(out_channels, in_channels, 4).normal_(0, 0.02)
        self.register_parameter('weight', nn.Parameter(weight))
        self.conv_op = flex_conv_op(ks=ks, stride=stride)

        if padding == 'same': 
            pad_ = (ks - 1) // 2
            self.pad_op  = nn.ZeroPad2d((pad_,  # pad left
                                         pad_,  # pad right
                                         pad_,  # pad top
                                         pad_)) # pad down 
        elif padding == 'round':
            pad_ = (ks - 1) // 2
            top_bot_pad = nn.ZeroPad2d((0, 0, pad_, pad_))
            side_pad    = lambda x : torch.cat([x[:, :, :, -pad_:], x, x[:, :, :, :pad_]], dim=-1)
            self.pad_op = lambda x : top_bot_pad(side_pad(x))

        else: 
            self.pad_op = lambda x : x
            

    def forward(self, x, points=None):
        if points is None:
            points = x[:, -3:] # get last 3 channels 
            points = F.tanh(points) # fit to [-1, 1]
            x      = x[:, :-3]
            x      = torch.cat([x, torch.zeros_like(points)], dim=1)
    
        x = self.pad_op(x)
        points = self.pad_op(points)
    
        return self.conv_op(x, self.weight, points)


class reg_conv(nn.Module):
    """ regular convolution wrapper """

    def __init__(self, in_channels, out_channels, ks, padding='same', stride=1):
        '''
        padding options are : 
        same  : zero padding such that (H',W') == (H,W)
        None  : no padding
        round : same padding, but with features from opposite sides
        '''
        super(reg_conv, self).__init__()
        self.conv_op = nn.Conv2d(in_channels, out_channels, ks, stride=stride)

        if padding is None or ks == 1: 
            self.pad_op = lambda x : x
        elif padding == 'same': 
            pad_ = (ks - 1) // 2
            self.pad_op  = nn.ZeroPad2d((pad_,  # pad left
                                         pad_,  # pad right
                                         pad_,  # pad top
                                         pad_)) # pad down 
        elif padding == 'round':
            pad_ = (ks - 1) // 2
            top_bot_pad = nn.ZeroPad2d((0, 0, pad_, pad_))
            side_pad    = lambda x : torch.cat([x[:, :, :, -pad_:], x, x[:, :, :, :pad_]], dim=-1)
            self.pad_op = lambda x : top_bot_pad(side_pad(x))


    def forward(self, x):
        return self.conv_op(self.pad_op(x))


class conv_res_block(nn.Module):
    
    """ conv + followed by  1x1 convolutions """
    def __init__(self, channels_in, 
                       channels_out, 
                       kernel_size, 
                       conv_type='flex',
                       padding='same', 
                       use_bn=True, 
                       act=nn.ReLU(), 
                       n_blocks=2, 
                       stride=1, 
                       fg_bias=0.5):
        
        super(conv_res_block, self).__init__()
        self.stride = stride
        self.fg_bias = fg_bias
        layers = [] 
        
        if not isinstance(use_bn, list):
            use_bn = [use_bn] * n_blocks
        if not isinstance(act, list):
            act    = [act]    * n_blocks

        for i in range(n_blocks):
            if i == 0: 
                ch_in = channels_in
                st = self.stride
                conv_op = flex_conv if conv_type == 'flex' else reg_conv
                ks = kernel_size
            else: 
                ch_in = channels_out
                st = 1
                conv_op = reg_conv
                ks = 1 # ? 

            layers += [conv_op(ch_in, 
                               channels_out, 
                               ks, 
                               padding=padding, 
                               stride=st)]

            if use_bn[i]:
                layers += [nn.BatchNorm2d(channels_out)]

            if act[i] is not None:
                layers += [act[i]]
        
        if channels_in != channels_out and fg_bias != 0: 
            self.merge = nn.Conv2d(channels_in, channels_out, 1)
        else: 
            self.merge = lambda x : x

        self.main = nn.ModuleList(layers)


    def forward(self, input, points=None):
        if points is None:
            output = self.main[0](input)
        else: 
            output = self.main[0](input, points)

        for i in range(1, len(self.main)):
            output = self.main[i](output)

        if self.fg_bias == 0.: 
            return output

        if self.stride != 1: 
            input = F.interpolate(input, scale_factor=1. / self.stride, mode='bilinear')
        
        input = self.merge(input)

        return input + self.fg_bias * output
        
        
# -------------------------------------------------------------------------------------------
# test flex conv implementation
# -------------------------------------------------------------------------------------------
def test_flex_conv_op():
    bs = np.random.randint(1, 15)
    i  = np.random.randint(1, 100)
    o  = np.random.randint(1, 100)
    ks = 2 * np.random.randint(1, 4) + 1 # must be odd 
    h  = np.random.randint(ks + 3, 60)
    w  = np.random.randint(ks + 3, 100)

    print('bs : {}\ni : {}\no : {}\nks : {}\nh : {}\nw : {}\n'.format(
            bs, i, o, ks, h, w))

    x = torch.randn(bs, i, h, w).cuda().normal_().double()
    kernel = torch.randn(o, i, 4).normal_().cuda().double()
    points = torch.randn(bs, 3, h, w).normal_().cuda().double()

    H, W, KS = h, w, ks

    out = flex_conv_op(ks=KS)(x, kernel, points)

    i = np.random.randint(KS // 2, H - KS + 1)
    j = np.random.randint(KS // 2, W - KS + 1)

    # let's run through one manually. 
    rel_pts = points[:, :, i, j].unsqueeze(-1).unsqueeze(-1) - \
                points

    # put the D axis fist, to do tensordot
    rel_pts_t = rel_pts.transpose(1,0)
    
    # calculate the weights for the specific point
    rel_weights = torch.einsum('oid,dbkw->oibkw', (kernel[:, :, :3], rel_pts_t))

    # add the bias to the weights
    rel_weights += kernel[:, :, -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # put batch first
    rel_weights = rel_weights.transpose(2,0).contiguous() # biokw

    x = x.unsqueeze(2) # bihw --> bi1hw

    # sum over the neighbours and I
    left = max(0, i - KS // 2)
    right = min(H, i + KS // 2 + 1)
    down = max(0, j - KS // 2)
    up = min(W, j + KS // 2 + 1)
    value = (rel_weights[:, :, :, left:right:, down:up] * 
            x[:, :, :, left:right, down:up])
    value = value.sum(1).sum(-1).sum(-1)     
    
    diff = (value - out[:, :, left, down]).abs()
    print('max error : {:.8f}\navg error : {:.8f}\ntotal nbs : {}'\
            .format(diff.max().item(),
                    diff.mean().item(), 
                    np.prod(diff.size())))

    if diff.max().item() > 1e-7: 
        raise ValueError('error too big')

def test_flex_conv_padding():
    bs = np.random.randint(1, 15)
    i  = np.random.randint(1, 100)
    o  = np.random.randint(1, 100)
    ks = 2 * np.random.randint(1, 4) + 1 # must be odd 
    h  = np.random.randint(ks + 3, 60)
    w  = np.random.randint(ks + 3, 100)
    st = np.random.randint(1, 3)

    h, w = st * h, st * w

    padding = ['same', 'round'][np.random.randint(0, 2)]

    print('bs : {}\ni : {}\no : {}\nks : {}\nh : {}\nw : {}\npad: {}\n st: {}'.format(
            bs, i, o, ks, h, w, padding, st))

    x = torch.randn(bs, i, h, w).cuda().normal_()
    points = torch.randn(bs, 3, h, w).normal_().cuda()
    out = flex_conv(i, o, ks, padding=padding, stride=st).cuda()(x, points)

    if sum([int(a) *st != int(b) for (a,b) in zip(out.shape[-2:], x.shape[-2:])]) > 0:
        raise ValueError('H and W should be equal')


if __name__ == '__main__':
    for _ in range(100):
        with torch.no_grad():
            test_flex_conv_op()
            test_flex_conv_padding()

