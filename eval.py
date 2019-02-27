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
    3) metric to evaluate on 
e.g. python eval.py runs/test_baseline 149 emd
'''

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
out_dir = os.path.join(sys.argv[1], 'final_samples')
maybe_create_dir(out_dir)
save_test_dataset = True
size = 5

# fetch metric
if 'emd' in sys.argv[3]: 
    loss = EMD
elif 'chamfer' in sys.argv[3]:
    loss = get_chamfer_dist
else:
    raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
            .format(sys.argv[2]))

with torch.no_grad():

    # 1) load trained model
    model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    model = model.cuda()
    model.eval()
    #model.args.no_polar = 1
    
    # 2) load data
    print('test set reconstruction')
    dataset = np.load('../lidar_generation/kitti_data/lidar_test.npz')#[:100]
    
    dataset = preprocess(dataset).astype('float32')

    if save_test_dataset: 
        np.save(os.path.join(out_dir, 'test_set'), dataset)

    dataset_test = from_polar_np(dataset) if model.args.no_polar else dataset
    loader = (torch.utils.data.DataLoader(dataset_test, batch_size=size,
                        shuffle=True, num_workers=4, drop_last=True)) #False))

    loss_fn = loss()
    
    # noisy reconstruction
    for noise in [0]:#, 0.15, 0.3]: #0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        losses = []
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
    
            losses += [loss_fn(recon_xyz, batch_xyz)]

        losses = torch.stack(losses).mean().item()
        print('{} with noise {} : {:.4f}'.format(sys.argv[3], noise, losses))

        del input, recon, recon_xyz, losses


    # missing reconstruction
    for missing in []:#0., 0.15, 0.3]: #, 0.25, 0.5, 0.75, 0.9]:
        losses = []
        for batch in loader:
            batch = batch.cuda()
            is_present = (torch.zeros_like(batch[:, [0]]).uniform_(0,1) + (1 - missing)).floor()
            input = batch * is_present
        
            recon = model(input)[0]
            recon_xyz = recon if model.args.no_polar else from_polar(recon)
            batch_xyz = batch if model.args.no_polar else from_polar(batch)

            # TODO: remove this
            recon_xyz.uniform_(batch_xyz.min(), batch_xyz.max())

            losses += [loss_fn(recon_xyz, batch_xyz)]
        
        losses = torch.stack(losses).mean().item()
        print('{} with missing p {} : {:.4f}'.format(sys.argv[3], missing, losses))

