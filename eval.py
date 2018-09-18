import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
import sys
from emd import EMD

from utils import * 
from models import * 

'''
Expect two arguments: 
    1) path_to_model_folder
    2) epoch of model you wish to load
e.g. python eval.py runs/test_baseline 149
'''

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
out_dir = os.path.join(sys.argv[1], 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = True
size = 10

with torch.no_grad():

    # 1) load trained model
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    model = model.cuda()
    model.eval()
    
    # 2) load data
    print('test set reconstruction')
    try:
        dataset = np.load('kitti_data/lidar_test.npz')
    except:
        dataset = np.load('../../lidar_generation/kitti_data/lidar_test.npz')
    
    # dataset = dataset[:50]
    dataset = preprocess(dataset).astype('float32')

    if save_test_dataset: 
        np.save(os.path.join(out_dir, 'test_set'), dataset)

    dataset_test = from_polar_np(dataset) if model.args.no_polar else dataset
    loader = (torch.utils.data.DataLoader(dataset_test, batch_size=size,
                        shuffle=True, num_workers=4, drop_last=False))


    emd_fn = EMD()
    
    # noisy reconstruction
    for noise in [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        emd = []
        for batch in loader:
            batch = batch.cuda()
            if not model.args.no_polar:
                batch = from_polar(batch)
        
            batch_xyz = batch
    
        
            input = batch_xyz + torch.zeros_like(batch_xyz).normal_(0, noise)
            if not model.args.no_polar:
                input = to_polar(input)
        
            input = input.cuda()
            recon = model(input)[0]

            recon_xyz = recon if model.args.no_polar else from_polar(recon)

            emd += [emd_fn(recon_xyz, batch_xyz)]

        emd = torch.stack(emd).mean().item()
        print('emd with noise {} : {:.4f}'.format(noise, emd))

        del input, recon, recon_xyz, emd


    # missing reconstruction
    for missing in [0.1]: #, 0.25, 0.5, 0.75, 0.9]:
        emd = []
        for batch in loader:
            batch = batch.cuda()
            is_present = (torch.zeros_like(batch[:, [0]]).uniform_(0,1) + (1 - missing)).floor()
            input = batch * is_present
        
            recon = model(input)[0]
            recon_xyz = recon if model.args.no_polar else from_polar(recon)
            batch_xyz = batch if model.args.no_polar else from_polar(batch)

            # TODO: remove this
            recon_xyz.uniform_(batch_xyz.min(), batch_xyz.max())

            emd += [emd_fn(recon_xyz, batch_xyz)]
        
        emd = torch.stack(emd).mean().item()
        print('emd with missing p {} : {:.4f}'.format(missing, emd))

