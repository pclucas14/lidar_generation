import numpy as np
import pdb
import os
import time
import sys



BASE_DIR='/mnt/data/lpagec/lidar_baseline'

# choose baseline
pb, ab = 0, 1
baseline = 'panos' if pb else 'atlas'

runs = 10
run_counter = 0
ran_exps = []
collisions = 0

while run_counter < runs:

    bs = [128, 64, 32]
    p = [0.2, 0.4, 0.4]
    bs = np.random.choice(bs, 1, p=p)[0]

    ## glr
    gen_lr = [1e-3, 5e-4]
    p = [0.5, 0.5]
    gen_lr = np.random.choice(gen_lr, 1, p=p)[0]

    # z size
    z_dim = [128, 256, 512]
    p = [1./3] * 3
    z_dim = np.random.choice(z_dim, 1, p=p)[0]

    base_dir = "%(BASE_DIR)s/%(baseline)s_Z%(z_dim)s_BS%(bs)s_GLR%(gen_lr)s" % locals()

    print(base_dir)

    command = "vae_2d.py \
        --base_dir %(base_dir)s \
        --lr %(gen_lr)s \
        --batch_size %(bs)s \
        --z_dim %(z_dim)s \
        --panos_baseline %(pb)s \
        --atlas_baseline %(ab)s " % locals()

    print(command)

    if 'mnt' in base_dir:
        command = ' python {}'.format(command)
    else:
        command = "{} cc_launch_gan.sh {}".format(sys.argv[1], command)
    print(command)

    if base_dir not in ran_exps:
        ran_exps += [base_dir]
        os.system(command)
        time.sleep(1)
        run_counter += 1
    else:
        print('run %s has already been launched! %d' % (base_dir, len(ran_exps)))
        collisions += 1

print('%d exps launched and %d collisions' % (runs, collisions))
