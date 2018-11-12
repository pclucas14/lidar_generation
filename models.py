import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
import pdb
from utils import *
try:
    import torch.nn.utils.spectral_norm as spectral_norm
except:
    print('pytorch version too old for spectral norm')
    spectral_norm = lambda x : x


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
# Core Models 
# --------------------------------------------------------------------------
class netG(nn.Module):
    def __init__(self, args, nz=100, ngf=64, nc=3, base=4, ff=(2,16)):
        super(netG, self).__init__()
        self.args = args
        conv = round_deconv2d if args.use_round_conv else nn.ConvTranspose2d
        sn   = spectral_norm if args.use_spectral_norm else lambda x : x

        layers  = []
        layers += [sn(conv(nz, ngf * 8, ff, 1, 0, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 8)] 
            layers += [nn.ReLU(True)]

        layers += [sn(conv(ngf * 8, ngf * 4, (3,4), stride=2, padding=(0,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 4)] 
            layers += [nn.ReLU(True)]
        
        layers += [sn(conv(ngf * 4, ngf * 2, (4,4), stride=2, padding=(1,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 2)] 
            layers += [nn.ReLU(True)]
        
        layers += [sn(conv(ngf * 2, ngf * 1, (4,4), stride=2, padding=(1,1), bias=False))]

        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ngf * 1)] 
            layers += [nn.ReLU(True)]

        layers += [sn(conv(ngf, nc, 4, 2, 1, bias=False))]
        layers += [nn.Tanh()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if len(input.shape) == 2: 
            input = input.unsqueeze(-1).unsqueeze(-1)
        
        return self.main(input)


class netD(nn.Module):
    def __init__(self, args, ndf=64, nc=2, nz=1, lf=(2,16)):
        super(netD, self).__init__()
        self.encoder = True if nz > 1 else False
        
        conv = round_conv2d if args.use_round_conv else nn.Conv2d
        sn   = spectral_norm if args.use_spectral_norm else (lambda x : x)

        layers  = []
        layers += [sn(conv(nc, ndf, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf, ndf * 2, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 2)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 4)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        
        layers += [sn(conv(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False))]
        
        if args.use_selu:
            layers += [nn.SELU()]
        else: 
            layers += [nn.BatchNorm2d(ndf * 8)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*layers)
        
        self.out = sn(conv(ndf * 8, nz, lf, 1, 0, bias=False))

    def forward(self, input, return_hidden=False):
        if input.size(-1) == 3: 
            input = input.transpose(1, 3)
        
        output_tmp = self.main(input)
        output = self.out(output_tmp)
       
        if return_hidden:
            return output, output_tmp
        
        return output if self.encoder else output.view(-1, 1).squeeze(1) 


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim, 
                                  AE=args.autoencoder, 
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim) 
        else: 
            mult = 1 if args.autoencoder else 2
            self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
            self.decode = netG(args, nz=args.z_dim, nc=3 if args.no_polar else 2)

        if not args.autoencoder and args.iaf:
            self.iaf = IAF(latent_size=args.z_dim)

    def forward(self, x):
        z = self.encode(x)
        while z.dim() != 2: 
            z = z.squeeze(-1)
            
        if self.args.autoencoder:
            return self.decode(z), None
        else:
            mu, logvar = torch.chunk(z, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd
            z = eps.mul(std).add_(mu) if self.training else mu

            if self.args.iaf:
                z_gaussian = z
                z, logdet = self.iaf(z_gaussian)
                kl = VAE.monte_carlo_kl(z=z, z_gauss=z_gaussian, log_det=logdet, posterior=[mu, std])
            else:
                kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def monte_carlo_kl(**kwargs):

        log_p_z_x = VAE.log_gauss(kwargs['z_gauss'], kwargs['posterior']) - kwargs['log_det']

        if kwargs.get('prior') is None:
            kwargs['prior'] = [t.zeros(*kwargs['z'].size()).cuda(),
                               t.ones(*kwargs['z'].size()).cuda()]

        one = t.cuda.FloatTensor([1])

        log_p_z = VAE.log_gauss(kwargs['z'], kwargs['prior'])

        result = log_p_z_x - log_p_z
        return result

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1) 


# --------------------------------------------------------------------------
# Code for Inverse Autoregressive flow, taken from 
# https://github.com/kefirski/bdir_vae/blob/master/IAF/IAF.py
# --------------------------------------------------------------------------
import torch as t
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
import math

class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(t.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(t.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input.matmul(self.weight.tril(-1))
        if self.bias is not None:
            output += self.bias
        return output


class IAF(nn.Module):
    def __init__(self, latent_size):
        super(IAF, self).__init__()

        self.z_size = latent_size


        self.m = nn.Sequential(
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

        self.s = nn.Sequential(
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

    def forward(self, z):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return:  An float tensor with shape of [batch_size, z_size] 
                  and log det value of the IAF mapping Jacobian
        """
        input = z

        m = self.m(input)
        s = self.s(input)

        z = s.exp() * z + m

        log_det = s.sum(1)

        return z, log_det


# --------------------------------------------------------------------------
# Baseline (AtlasNet), taken from https://github.com/ThibaultGROUEIX/AtlasNet
# --------------------------------------------------------------------------
class PointNetfeat_(nn.Module):
    def __init__(self, num_points = 40 * 256, global_feat = True):
        super(PointNetfeat_, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 128):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 40 * 256, bottleneck_size = 1024, nb_primitives = 2, AE=True):
        super(AE_AtlasNet, self).__init__()
        bot_enc = bottleneck_size if AE else 2 * bottleneck_size
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat_(num_points, global_feat=True),
        nn.Linear(1024, bot_enc),
        nn.BatchNorm1d( bot_enc),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 + self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def encode(self, x):
        if x.dim() == 4 : 
            if x.size(1) != 3: 
                assert x.size(-1) == 3 
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.reshape(x.size(0), 3, -1)
        else: 
            if x.size(1) != 3: 
                assert x.size(-1) == 3 
                x = x.transpose(-1, -2).contiguous()
        
        x = self.encoder(x)
        return x

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = (torch.cuda.FloatTensor(x.size(0),2,self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()



if __name__ == '__main__':
    points = torch.cuda.FloatTensor(10, 3, 40, 256).normal_()
    AE = AE_AtlasNet(num_points = 40 * 256).cuda()
    out = AE(points)
    loss = get_chamfer_dist()(points, out)
    x =1


# --------------------------------------------------------------------------
# Baseline (Panos's paper)
# --------------------------------------------------------------------------
class PointGenPSG2(nn.Module):
    def __init__(self, nz=100, num_points = 40 * 256):
        super(PointGenPSG2, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3 // 2)

        self.fc11 = nn.Linear(nz, 256)
        self.fc21 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(512, 1024)
        self.fc41 = nn.Linear(1024, self.num_points * 3 // 2)
        self.th = nn.Tanh()
        self.nz = nz
        
    
    def forward(self, x):
        batchsize = x.size()[0]
        
        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)

        x2 = F.relu(self.fc11(x2))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc31(x2))
        x2 = self.th(self.fc41(x2))
        x2 = x2.view(batchsize, 3, -1)

        return torch.cat([x1, x2], 2)

