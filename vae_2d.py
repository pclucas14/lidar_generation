import comet_ml
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torchvision import datasets, transforms
from comet_ml import Experiment
from pydoc import locate
import tensorboardX

from utils import * 
from models import * 

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--use_selu', type=int, default=0, help='replaces batch_norm + act with SELU')
parser.add_argument('--use_spectral_norm', type=int, default=0)
parser.add_argument('--use_round_conv', type=int, default=0)
parser.add_argument('--base_dir', type=str, default='runs/test')
parser.add_argument('--no_polar', type=int, default=0)
parser.add_argument('--optim',  type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--iaf', type=int, default=0)
parser.add_argument('--autoencoder', type=int, default=0)
parser.add_argument('--atlas_baseline', type=int, default=0, help='this flag is also used to determine the number of primitives used in the model')
parser.add_argument('--panos_baseline', type=int, default=0)
parser.add_argument('--kl_warmup_epochs', type=int, default=150)
parser.add_argument('--save_samples', type=int, default=0)
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--test', action='store_true')
parser.add_argument('--disc', type=str, default='conv', choices=['conv', 'flex'])
parser.add_argument('--attention', type=int, default=0)

args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# construct model and ship to GPU
model = VAE(args).cuda()

# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
if args.log: 
    experiment = Experiment(api_key='WXSjTPlJVTQlUBN2o3O5D6Pwz', 
                            project_name='lidar-generation', 
                            workspace='pclucas14')
    experiment.log_parameters(vars(args))

writes = 0
ns     = 16

# dataset preprocessing
dataset = np.load('kitti_data/lidar.npz')[:1000]
dataset = preprocess(dataset).astype('float32')
dataset_train = from_polar_np(dataset) if args.no_polar else dataset

dataset = np.load('kitti_data/lidar_val.npz') 
dataset = preprocess(dataset).astype('float32')
dataset_val = from_polar_np(dataset) if args.no_polar else dataset

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, drop_last=False)

print(model)
print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr) 
# construction reconstruction loss function
if args.atlas_baseline or args.panos_baseline:
    loss_fn = get_chamfer_dist()
else:
    loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1) 



# VAE training
# ------------------------------------------------------------------------------------------------
for epoch in range(500):
    print('epoch %s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    for i, img in enumerate(train_loader):
        img = img.cuda()
        recon, kl_cost = model(img)

        loss_recon = loss_fn(recon, img)
        kl_obj     = min(1, float(epoch+1) / args.kl_warmup_epochs) * torch.clamp(kl_cost, min=5)
        
        loss = (kl_obj + loss_recon).mean(dim=0)
        elbo = (kl_cost + loss_recon).mean(dim=0)

        loss_    += [loss.item()]
        elbo_    += [elbo.item()]
        kl_cost_ += [kl_cost.mean(dim=0).item()]
        kl_obj_  += [kl_obj.mean(dim=0).item()]
        recon_   += [loss_recon.mean(dim=0).item()]

        # baseline loss is very memory heavy 
        # we accumulate gradient to simulate a bigger minibatch
        if (i+1) % 4 == 0 or not args.atlas_baseline: 
            optim.zero_grad()

        loss.backward()
        if (i+1) % 4 == 0 or not args.atlas_baseline: 
            optim.step()

    writes += 1
    mn = lambda x : np.mean(x)
    if args.log:
        experiment.log_metric('train/loss', mn(loss_), step=writes)
        experiment.log_metric('train/elbo', mn(elbo_), step=writes)
        experiment.log_metric('train/kl_cost', mn(kl_cost_), step=writes)
        experiment.log_metric('train/kl_obj', mn(kl_obj_), step=writes)
        experiment.log_metric('train/recon', mn(recon_), writes)
    
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
        
    # save some training reconstructions
    if args.save_samples:
        recon = recon[:ns].cpu().data.numpy()
        with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
            np.save(f, recon)

    # log_point_clouds(writer, recon[:ns], 'train_recon', step=writes)
    print('saved training reconstructions')
    
    
    # Testing loop
    # ----------------------------------------------------------------------

    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            print('test set evaluation')
            for i, img in enumerate(val_loader):
                img = img.cuda()
                recon, kl_cost = model(img)
            
                loss_recon = loss_fn(recon, img) 
                kl_obj     =  min(1, float(epoch) / args.kl_warmup_epochs) * torch.clamp(kl_cost, min=5)

                loss = (kl_obj + loss_recon).mean(dim=0)
                elbo = (kl_cost + loss_recon).mean(dim=0)

                loss_    += [loss.item()]
                elbo_    += [elbo.item()]
                kl_cost_ += [kl_cost.mean(dim=0).item()]
                kl_obj_  += [kl_obj.mean(dim=0).item()]
                recon_   += [loss_recon.mean(dim=0).item()]

                # if epoch % 5 != 0 and i > 5 : break
            
            if args.log:
                experiment.log_metric('valid/loss', mn(loss_), step=writes)
                experiment.log_metric('valid/elbo', mn(elbo_), step=writes)
                experiment.log_metric('valid/kl_cost', mn(kl_cost_), step=writes)
                experiment.log_metric('valid/kl_obj', mn(kl_obj_), step=writes)
                experiment.log_metric('valid/recon', mn(recon_), writes)
            
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

            with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
                recon = recon[:ns].cpu().data.numpy()
                np.save(f, recon)
                print('saved test recons')

           
            if args.save_samples:
                sample = model.sample()
                with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
                    sample = sample.cpu().data.numpy()
                    np.save(f, recon)
                
                #log_point_clouds(writer, sample, 'vae_samples', step=writes)
                print('saved model samples')

                if epoch == 0: 
                    with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
                        img = img.cpu().data.numpy()
                        np.save(f, img)
                    
                    # log_point_clouds(writer, img[:ns], 'real_lidar', step=writes)
                    print('saved real LiDAR')
                
    
    if (epoch + 1) % 10 == 0 :
        torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
