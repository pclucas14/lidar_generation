import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import sys

import __init__
from utils import * 
from models import * 

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
out_dir = sys.argv[3] #os.path.join(sys.argv[3], 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = True


with torch.no_grad():
    '''
    # 1 ) unconditional generation
    print('unconditional generation')
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    model = model.cuda()
    samples = []
    
    with torch.no_grad():
        try:
            for temp in [0.2, 0.5, 0.7, 1]:
                z_ = model.args.z_dim
                is_vae = True
                model.eval()
                out = model.sample(nb_samples=nb_samples)
                np.save(os.path.join(out_dir, 'uncond_{}'.format(temp)), out.cpu().data.numpy())
        except:
            z_ = 100
            noise = torch.cuda.FloatTensor(nb_samples, z_).normal_()
            out = model(noise)
            is_vae = False
            np.save(os.path.join(out_dir, 'uncond'), out.cpu().data.numpy())


    # 2) undonditional interpolation
    print('unconditional interpolation')
    noise = []
    for _ in range(100):
        noise += [torch.cuda.FloatTensor(1, z_).normal_()]

    noises = []

    slices = 8
    for i in range(len(noise) - 1):
        noises += [noise[i]]
        for j in range(slices):
            alpha = float(j) / slices
            noises += [noise[i] * (1. - alpha) + noise[i+1] * alpha]
    
    noises += [noise[-1]]
    noises = torch.cat(noises, dim=0)
    

    if is_vae:
        out = model.decode(noises)
    else:
        out = model(noises)

    np.save(os.path.join(out_dir, 'undond_inter'), out.cpu().data.numpy())
    
    if not is_vae:
        exit()
    
    '''
    # 3) test set reconstruction
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0].cuda()
    print('test set reconstruction')
    try:
        dataset = np.load('kitti_data/lidar_test.npz')
    except:
        dataset = np.load('../lidar_generation/kitti_data/lidar_test.npz')
    dataset = preprocess(dataset).astype('float32')

    if save_test_dataset: 
        np.save(os.path.join(out_dir, 'test_set'), dataset)

    # dataset_test = from_polar_np(dataset) if model.args.no_polar else dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                        shuffle=False, num_workers=4, drop_last=False)

    '''
    samples = []
    real = []
    for real_data in loader:
        #print(real_data[0].mean())
        real_data = real_data.cuda()
        out = model(real_data)[0].cpu().data
        samples += [out]
        real += [real_data]
    

    samples = torch.cat(samples, dim=0).cpu().data.numpy()#.transpose(0, 2, 1)
    real    = torch.cat(real, dim=0).cpu().data.numpy()

    if not model.args.no_polar:
        samples, real = from_polar_np(samples), from_polar_np(real)

    for i in range(samples.shape[0]):
        show_pc(samples[i], save_path=os.path.join(out_dir, 'ATLASsample_%d.png' % i))
        #show_pc(real[i],    save_path=os.path.join(out_dir,   'ATLASreal_%d.png' % i))
    
    
    exit()
    np.save(os.path.join(out_dir, 'recon'), samples.cpu().data.numpy())
    np.save(os.path.join(out_dir, 'real'), real.cpu().data.numpy())
    
    real_data = next(loader).cuda()
    out = model(real_data)[0]
    np.save(os.path.join(out_dir, 'recon'), out.cpu().data.numpy())

    print('test set interpolation')
    aa, bb = real_data, next(loader).cuda()

    noise_a, noise_b = model.encode(aa).chunk(2, dim=1)[0], model.encode(bb).chunk(2, dim=1)[0]
    alpha  = np.arange(10) / 10.
    noises, out = [], []
    for a in alpha:
        noises += [a * noise_a + (1 - a) * noise_b]

    for noise in noises:
        noise = noise.cuda()
        out += [model.decode(noise)]

    out = torch.stack(out, dim=1)
    # add ground truth to saved tensors
    if model.args.no_polar and not model.args.atlas_baseline:
        shp = out.shape
        out = to_polar(out.reshape(shp[0] * shp[1], shp[2], shp[3], shp[4]))
        out = out.reshape(*shp)
        a = 1

    out = torch.cat([bb.unsqueeze(1), out, aa.unsqueeze(1)], dim=1)
    for i, inter in enumerate(out[:100]):
        np.save(os.path.join(out_dir, 'cond_inter_%d' % i), inter.cpu().data.numpy())

    '''
    print('noisy reconstruction')
    for noise in [0, 0.3, 0.7, 1.]:
        reals, recons, corrs = [], [], []

        process_inp = (lambda x : x) if model.args.no_polar else to_polar
        for batch in loader:

            batch = batch.cuda()
            batch_xyz = from_polar(batch)
            noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)

            means = batch_xyz.transpose(1,0).reshape((3, -1)).mean(dim=-1)
            stds  = batch_xyz.transpose(1,0).reshape((3, -1)).std(dim=-1)
            means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]

            # normalize data
            norm_batch_xyz = (batch_xyz - means) / (stds + 1e-9)
            # add the noise
            input = norm_batch_xyz + noise_tensor

            # unnormalize
            input = input * (stds + 1e-9) + means

            recon = model(process_inp(input))[0]

            recons += [recon]
            reals  += [batch_xyz]
            corrs  += [input]

            '''
            ind = -1
            show = show_pc_lite
            reals = batch_xyz[ind].reshape(3, -1).transpose(1,0)
            fakes = from_polar(recon)[ind].reshape(3, -1).transpose(1,0)
            corr  = from_polar(input)[ind].reshape(3, -1).transpose(1,0)
            import pdb; pdb.set_trace()
            last = 'line'
            break

            '''

        ps = lambda x : x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 3)
        reals  = ps(torch.cat(reals))
        corrs  = ps(torch.cat(corrs))
        recons = torch.cat(recons).transpose(-2, -1)# 
        #recons = ps(torch.cat(recons))

        for name, arr in zip(['real', 'corr', 'recon'], [reals, corrs, recons]):
            np.save(os.path.join(out_dir, '{}_{:.4f}'.format(name, noise)), arr)

        ''' 
        if model.args.no_polar and not model.args.atlas_baseline:
            shp = out.shape
            out = to_polar(out.reshape(shp[0] * shp[1], shp[2], shp[3], shp[4]))
            out = out.reshape(*shp)

        out = out.reshape(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
        np.save(os.path.join(out_dir, 'noisy_recon_{}'.format(noise)), out.cpu().data.numpy())
        '''


    '''
    print('corrupted reconstruction')
    for missing in [0.1, 0.25, 0.5, 0.75, 0.9]:
        is_present = (torch.zeros_like(real_data[:, [0]]).uniform_(0,1) + (1 - missing)).floor()
        input = real_data * is_present
        recon = model(input)[0]

        out = torch.stack([real_data, recon, input], dim=1)
        out = out.reshape(out.size(0) * out.size(1), out.size(2), out.size(3), out.size(4))
        np.save(os.path.join(out_dir, 'missing_recon_{}'.format(missing)), out.cpu().data.numpy())
    ''' 
