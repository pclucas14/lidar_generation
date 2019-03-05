from torch.autograd import Function, Variable
import numpy as np
import torch
import torch.nn as nn
import pdb
from torch.nn.modules.utils import _pair
from pyinn.utils import Dtype, Stream, load_kernel
import torch.nn.functional as F
from torch.autograd import gradcheck



approxmatch_kernel = '''
extern "C"
__global__ void approxmatch(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,float * __restrict__ match,float * temp){
        float * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
        float multiL,multiR;
        if (n>=m){
                multiL=1;
                multiR=n/m;
        }else{
                multiL=m/n;
                multiR=1;
        }
        const int Block=1024;
        __shared__ float buf[Block*4];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
                        match[i*n*m+j]=0;
                for (int j=threadIdx.x;j<n;j+=blockDim.x)
                        remainL[j]=multiL;
                for (int j=threadIdx.x;j<m;j+=blockDim.x)
                        remainR[j]=multiR;
                __syncthreads();
                for (int j=7;j>=-2;j--){
                        float level=-powf(4.0f,j);
                        if (j==-2){
                                level=0;
                        }
                        for (int k0=0;k0<n;k0+=blockDim.x){
                                int k=k0+threadIdx.x;
                                float x1=0,y1=0,z1=0;
                                if (k<n){
                                        x1=xyz1[i*n*3+k*3+0];
                                        y1=xyz1[i*n*3+k*3+1];
                                        z1=xyz1[i*n*3+k*3+2];
                                }
                                float suml=1e-9f;
                                for (int l0=0;l0<m;l0+=Block){
                                        int lend=min(m,l0+Block)-l0;
                                        for (int l=threadIdx.x;l<lend;l+=blockDim.x){
                                                float x2=xyz2[i*m*3+l0*3+l*3+0];
                                                float y2=xyz2[i*m*3+l0*3+l*3+1];
                                                float z2=xyz2[i*m*3+l0*3+l*3+2];
                                                buf[l*4+0]=x2;
                                                buf[l*4+1]=y2;
                                                buf[l*4+2]=z2;
                                                buf[l*4+3]=remainR[l0+l];
                                        }
                                        __syncthreads();
                                        for (int l=0;l<lend;l++){
                                                float x2=buf[l*4+0];
                                                float y2=buf[l*4+1];
                                                float z2=buf[l*4+2];
                                                float d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
                                                float w=__expf(d)*buf[l*4+3];
                                                suml+=w;
                                        }
                                        __syncthreads();
                                }
                                if (k<n)
                                        ratioL[k]=remainL[k]/suml;
                        }
                        /*for (int k=threadIdx.x;k<n;k+=gridDim.x){
                                float x1=xyz1[i*n*3+k*3+0];
                                float y1=xyz1[i*n*3+k*3+1];
                                float z1=xyz1[i*n*3+k*3+2];
                                float suml=1e-9f;
                                for (int l=0;l<m;l++){
                                        float x2=xyz2[i*m*3+l*3+0];
                                        float y2=xyz2[i*m*3+l*3+1];
                                        float z2=xyz2[i*m*3+l*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*remainR[l];
                                        suml+=w;
                                }
                                ratioL[k]=remainL[k]/suml;
                        }*/
                        __syncthreads();
                        for (int l0=0;l0<m;l0+=blockDim.x){
                                int l=l0+threadIdx.x;
                                float x2=0,y2=0,z2=0;
                                if (l<m){
                                        x2=xyz2[i*m*3+l*3+0];
                                        y2=xyz2[i*m*3+l*3+1];
                                        z2=xyz2[i*m*3+l*3+2];
                                }
                                float sumr=0;
                                for (int k0=0;k0<n;k0+=Block){
                                        int kend=min(n,k0+Block)-k0;
                                        for (int k=threadIdx.x;k<kend;k+=blockDim.x){
                                                buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
                                                buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
                                                buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
                                                buf[k*4+3]=ratioL[k0+k];
                                        }
                                        __syncthreads();
                                        for (int k=0;k<kend;k++){
                                                float x1=buf[k*4+0];
                                                float y1=buf[k*4+1];
                                                float z1=buf[k*4+2];
                                                float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*buf[k*4+3];
                                                sumr+=w;
                                        }
                                        __syncthreads();
                                }
                                if (l<m){
                                        sumr*=remainR[l];
                                        float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
                                        ratioR[l]=consumption*remainR[l];
                                        remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
                                }
                        }
                        /*for (int l=threadIdx.x;l<m;l+=blockDim.x){
                                float x2=xyz2[i*m*3+l*3+0];
                                float y2=xyz2[i*m*3+l*3+1];
                                float z2=xyz2[i*m*3+l*3+2];
                                float sumr=0;
                                for (int k=0;k<n;k++){
                                        float x1=xyz1[i*n*3+k*3+0];
                                        float y1=xyz1[i*n*3+k*3+1];
                                        float z1=xyz1[i*n*3+k*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k];
                                        sumr+=w;
                                }
                                sumr*=remainR[l];
                                float consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
                                ratioR[l]=consumption*remainR[l];
                                remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
                        }*/
                        __syncthreads();
                        for (int k0=0;k0<n;k0+=blockDim.x){
                                int k=k0+threadIdx.x;
                                float x1=0,y1=0,z1=0;
                                if (k<n){
                                        x1=xyz1[i*n*3+k*3+0];
                                        y1=xyz1[i*n*3+k*3+1];
                                        z1=xyz1[i*n*3+k*3+2];
                                }
                                float suml=0;
                                for (int l0=0;l0<m;l0+=Block){
                                        int lend=min(m,l0+Block)-l0;
                                        for (int l=threadIdx.x;l<lend;l+=blockDim.x){
                                                buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
                                                buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
                                                buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
                                                buf[l*4+3]=ratioR[l0+l];
                                        }
                                        __syncthreads();
                                        float rl=ratioL[k];
                                        if (k<n){
                                                for (int l=0;l<lend;l++){
                                                        float x2=buf[l*4+0];
                                                        float y2=buf[l*4+1];
                                                        float z2=buf[l*4+2];
                                                        float w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*rl*buf[l*4+3];
                                                        match[i*n*m+(l0+l)*n+k]+=w;
                                                        suml+=w;
                                                }
                                        }
                                        __syncthreads();
                                }
                                if (k<n)
                                        remainL[k]=fmaxf(0.0f,remainL[k]-suml);
                        }
                        /*for (int k=threadIdx.x;k<n;k+=blockDim.x){
                                float x1=xyz1[i*n*3+k*3+0];
                                float y1=xyz1[i*n*3+k*3+1];
                                float z1=xyz1[i*n*3+k*3+2];
                                float suml=0;
                                for (int l=0;l<m;l++){
                                        float x2=xyz2[i*m*3+l*3+0];
                                        float y2=xyz2[i*m*3+l*3+1];
                                        float z2=xyz2[i*m*3+l*3+2];
                                        float w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k]*ratioR[l];
                                        match[i*n*m+l*n+k]+=w;
                                        suml+=w;
                                }
                                remainL[k]=fmaxf(0.0f,remainL[k]-suml);
                        }*/
                        __syncthreads();
                }
        }
}'''

matchcost_kernel = '''
extern "C"
__global__ void matchcost(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ out){
        __shared__ float allsum[512];
        const int Block=1024;
        __shared__ float buf[Block*3];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                float subsum=0;
                for (int k0=0;k0<n;k0+=blockDim.x){
                        int k=k0+threadIdx.x;
                        float x1=0,y1=0,z1=0;
                        if (k<n){
                                x1=xyz1[i*n*3+k*3+0];
                                y1=xyz1[i*n*3+k*3+1];
                                z1=xyz1[i*n*3+k*3+2];
                        }
                        for (int l0=0;l0<m;l0+=Block){
                                int lend=min(m,l0+Block)-l0;
                                for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
                                        buf[l]=xyz2[i*m*3+l0*3+l];
                                __syncthreads();
                                if (k<n){
                                        for (int l=0;l<lend;l++){
                                                float x2=buf[l*3+0];
                                                float y2=buf[l*3+1];
                                                float z2=buf[l*3+2];
                                                float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
                                                subsum+=d*match[i*n*m+(l0+l)*n+k];
                                        }
                                }
                                __syncthreads();
                        }
                }
                allsum[threadIdx.x]=subsum;
                for (int j=1;j<blockDim.x;j<<=1){
                        __syncthreads();
                        if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
                                allsum[threadIdx.x]+=allsum[threadIdx.x+j];
                        }
                }
                if (threadIdx.x==0)
                        out[i]=allsum[0];
                __syncthreads();
        }
}'''

matchcostgrad2_kernel = '''
extern "C"
__global__ void matchcostgrad2(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad2){
        __shared__ float sum_grad[256*3];
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                int kbeg=m*blockIdx.y/gridDim.y;
                int kend=m*(blockIdx.y+1)/gridDim.y;
                for (int k=kbeg;k<kend;k++){
                        float x2=xyz2[(i*m+k)*3+0];
                        float y2=xyz2[(i*m+k)*3+1];
                        float z2=xyz2[(i*m+k)*3+2];
                        float subsumx=0,subsumy=0,subsumz=0;
                        for (int j=threadIdx.x;j<n;j+=blockDim.x){
                                float x1=x2-xyz1[(i*n+j)*3+0];
                                float y1=y2-xyz1[(i*n+j)*3+1];
                                float z1=z2-xyz1[(i*n+j)*3+2];
                                float d=match[i*n*m+k*n+j]*rsqrtf(fmaxf(x1*x1+y1*y1+z1*z1,1e-20f));
                                subsumx+=x1*d;
                                subsumy+=y1*d;
                                subsumz+=z1*d;
                        }
                        sum_grad[threadIdx.x*3+0]=subsumx;
                        sum_grad[threadIdx.x*3+1]=subsumy;
                        sum_grad[threadIdx.x*3+2]=subsumz;
                        for (int j=1;j<blockDim.x;j<<=1){
                                __syncthreads();
                                int j1=threadIdx.x;
                                int j2=threadIdx.x+j;
                                if ((j1&j)==0 && j2<blockDim.x){
                                        sum_grad[j1*3+0]+=sum_grad[j2*3+0];
                                        sum_grad[j1*3+1]+=sum_grad[j2*3+1];
                                        sum_grad[j1*3+2]+=sum_grad[j2*3+2];
                                }
                        }
                        if (threadIdx.x==0){
                                grad2[(i*m+k)*3+0]=sum_grad[0];
                                grad2[(i*m+k)*3+1]=sum_grad[1];
                                grad2[(i*m+k)*3+2]=sum_grad[2];
                        }
                        __syncthreads();
                }
        }
}'''

matchcostgrad1_kernel = '''
extern "C"
__global__ void matchcostgrad1(int b,int n,int m,const float * __restrict__ xyz1,const float * __restrict__ xyz2,const float * __restrict__ match,float * __restrict__ grad1){
        for (int i=blockIdx.x;i<b;i+=gridDim.x){
                for (int l=threadIdx.x;l<n;l+=blockDim.x){
                        float x1=xyz1[i*n*3+l*3+0];
                        float y1=xyz1[i*n*3+l*3+1];
                        float z1=xyz1[i*n*3+l*3+2];
                        float dx=0,dy=0,dz=0;
                        for (int k=0;k<m;k++){
                                float x2=xyz2[i*m*3+k*3+0];
                                float y2=xyz2[i*m*3+k*3+1];
                                float z2=xyz2[i*m*3+k*3+2];
                                float d=match[i*n*m+k*n+l]*rsqrtf(fmaxf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2),1e-20f));
                                dx+=(x1-x2)*d;
                                dy+=(y1-y2)*d;
                                dz+=(z1-z2)*d;
                        }
                        grad1[i*n*3+l*3+0]=dx;
                        grad1[i*n*3+l*3+1]=dy;
                        grad1[i*n*3+l*3+2]=dz;
                }
        }
}
'''
#void approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp){
#       approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
# }



class EMD(Function):

    def forward(self, a, b):

        # same preprocessing as chamfer 
        if a.dim() == 4:
            if a.size(1) == 2: 
                a = from_polar(a)

            assert a.size(1) == 3
            a = a.permute(0, 2, 3, 1).contiguous().reshape(a.size(0), -1, 3)
            
        if b.dim() == 4:
            if b.size(1) == 2: 
                b = from_polar(b)

            assert b.size(1) == 3
            b = b.permute(0, 2, 3, 1).contiguous().reshape(b.size(0), -1, 3)

        assert a.dim() == b.dim() == 3
        if a.size(-1) != 3: 
            assert a.size(-2) == 3
            a = a.transpose(-2, -1).contiguous()
        
        if b.size(-1) != 3: 
            assert b.size(-2) == 3
            b = a.transpose(-2, -1).contiguous()

        xyz1, xyz2 = a, b
        batch_size, num_pts, pt_dim = xyz1.size()
        _         , m      , _      = xyz2.size()

        match = torch.zeros(batch_size, m, num_pts).cuda()
        cost = torch.zeros(batch_size, ).cuda()
        temp = torch.zeros(batch_size, 2 * (m+num_pts)).cuda()

        n = xyz1.numel()

        with torch.cuda.device_of(xyz1):
            # 1) get matching
            f = load_kernel('approxmatch', approxmatch_kernel)
            f(block=(512, 1, 1), # (CUDA_NUM_THREADS,1,1),
              grid=(32, 1, 1),   # GET_BLOCKS(n),1,1),
              args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(), match.data_ptr(), temp.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            # 2) calculate matching cost
            g = load_kernel('matchcost', matchcost_kernel)
            g(block=(512, 1, 1), # (CUDA_NUM_THREADS, 1, 1),
              grid=(32, 1, 1),   # (GET_BLOCKS(n), 1, 1),
              args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(), match.data_ptr(), cost.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


        self.xyz1       = xyz1
        self.xyz2       = xyz2
        self.match      = match
        self.cost       = cost
        self.num_pts    = num_pts
        self.m          = m
        self.numel      = n
        self.batch_size = batch_size
        del temp

        return cost

    def backward(self, grad_cost):
        xyz1       = self.xyz1
        xyz2       = self.xyz2
        match      = self.match
        cost       = self.cost
        m          = self.m
        numel      = self.numel
        num_pts    = self.num_pts
        batch_size = self.batch_size

        grad1 = torch.zeros_like(xyz1).cuda()
        grad2 = torch.zeros_like(xyz2).cuda()

        with torch.cuda.device_of(grad_cost):
            if xyz1.requires_grad:
                f = load_kernel('matchcostgrad1', matchcostgrad1_kernel)
                f(block=(512, 1, 1), # (CUDA_NUM_THREADS, 1, 1),
                grid=(32, 1, 1),     # (GET_BLOCKS(xyz1.numel()), 1, 1),
                args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(), match.data_ptr(), grad1.data_ptr()],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if xyz2.requires_grad:
                g = load_kernel('matchcostgrad2', matchcostgrad2_kernel)
                g(block=(256, 1, 1), # (CUDA_NUM_THREADS, 1, 1),
                grid=(32, 32, 1),    # (GET_BLOCKS(xyz2.numel()), 1, 1),
                args=[batch_size, num_pts, m, xyz1.data_ptr(), xyz2.data_ptr(), match.data_ptr(), grad2.data_ptr()],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad1 * grad_cost.view(-1, 1, 1), grad2 * grad_cost.view(-1, 1, 1)
