'''
Taken from https://github.com/nicola-decao/s-vae-pytorch
'''
import torch
import math
import numpy as np
import scipy.special
from numbers import Number
from torch.distributions.kl import register_kl

class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, v, z):
        
        assert isinstance(v, Number), 'v must be a scalar'
        
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        
        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else: #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
#         else:
#             print(v, type(v), np.isclose(v, 0))
#             raise RuntimeError('v must be >= 0, it is {}'.format(v))
        
        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z)

class Ive(torch.nn.Module):
    
    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v
        
    def forward(self, z):
        return ive(self.v, z)

# ive = IveFunction.apply

class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim
    
    def __init__(self, dim, validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim

    def sample(self, shape=torch.Size()):        
        output = torch.distributions.Normal(0, 1).sample(
            (shape if isinstance(shape, torch.Size) else torch.Size([shape])) + torch.Size([self._dim + 1]))

        return output / output.norm(dim=-1, keepdim=True)

    def sample_pair(self, num_samples, step_size=1.):
        sample_a = torch.cuda.FloatTensor(num_samples, self._dim).normal_()
        sample_b = torch.cuda.FloatTensor(num_samples, self._dim).normal_()
        sample_b = sample_b + sample_a
        return [x / x.norm(dim=-1, keepdim=True) for x in [sample_a, sample_b]]
        

    def entropy(self):
        return self.__log_surface_area()
    
    def log_prob(self, x):
        return - torch.ones(x.shape[:-1]) * self.__log_surface_area()

    def __log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
            torch.Tensor([(self._dim + 1) / 2]))


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    x = 1
