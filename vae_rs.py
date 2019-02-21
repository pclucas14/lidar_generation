import numpy as np
import pdb
import os
import time
import sys



BASE_DIR='/home/lpagec/scratch/lidar_generation/vae2'

runs = 3*3*2 
run_counter = 0
ran_exps = []
collisions = 0

while run_counter < runs:

    disc = ['base', 'flex', 'attn']
    p = [1./3] * 3
    disc_ = np.random.choice(disc, 1, p=p)[0]

    if disc_ == 'base':
        disc = 'conv'
        attention = 0 
    elif disc_ == 'flex':
        disc = 'flex'
        attention = 0
    elif disc_ == 'attn':
        attention = 1
        disc = 'conv'

    bs = [128, 64, 32]
    p = [0.2, 0.4, 0.4]
    bs = np.random.choice(bs, 1, p=p)[0]

    ## glr
    gen_lr = [1e-4, 2e-4]
    p = [0.5, 0.5]
    gen_lr = np.random.choice(gen_lr, 1, p=p)[0]

    base_dir = "%(BASE_DIR)s/DISC%(disc_)s_BS%(bs)s_GLR%(gen_lr)s" % locals()

    print(base_dir)

    command = "vae_2d.py \
        --disc %(disc)s \
        --attention %(attention)s \
        --base_dir %(base_dir)s \
        --lr %(gen_lr)s \
        --batch_size %(bs)s" % locals()

    print(command)

    command = "{} cc_launch_gan.sh {}".format(sys.argv[1], command)
    print(command)

    if base_dir not in ran_exps:
        ran_exps += [base_dir]
        os.system(command)
        time.sleep(2)
        run_counter += 1
    else:
        print('run %s has already been launched! %d' % (base_dir, len(ran_exps)))
        collisions += 1

print('%d exps launched and %d collisions' % (runs, collisions))
