import numpy as np
import pdb
import os
import time
import sys



BASE_DIR='/home/lpagec/scratch/lidar_generation/seq_gan'

runs = 25
run_counter = 0
ran_exps = []
collisions = 0

while run_counter < runs:

    loss = [0, 1]
    p = [0.7, 0.3]
    loss = np.random.choice(loss, 1, p=p)[0]

    sn = [0, 1]
    p = [0.5] * 2
    sn = np.random.choice(sn, 1, p=p)[0]

    bs = [128, 64, 32]
    p = [0.2, 0.4, 0.4]
    bs = np.random.choice(bs, 1, p=p)[0]
    bs = 32 

    ## glr
    gen_lr = [1e-4, 2e-4]
    p = [0.7, 0.3]
    gen_lr = np.random.choice(gen_lr, 1, p=p)[0]

    dis_lr = [1e-4, 2e-4]
    dis_lr = np.random.choice(dis_lr, 1, p=p)[0]

    opt = ['adam', 'rmsprop']
    p = [0.5] * 2
    opt = np.random.choice(opt, 1, p=p)[0]

    base_dir = "%(BASE_DIR)s/Loss%(loss)s_SN%(sn)s_BS%(bs)s_GLR%(gen_lr)s_DLR%(dis_lr)s" % locals()

    print(base_dir)

    command = "seq_gan.py \
        --loss %(loss)s \
        --base_dir %(base_dir)s \
        --gen_lr %(gen_lr)s \
        --dis_lr %(dis_lr)s \
        --use_spectral_norm %(sn)s \
        --optim %(opt)s \
        --batch_size %(bs)s" % locals()

    print(command)

    command = "{} cc_launch_gan.sh {}".format(sys.argv[1], command)

    if base_dir not in ran_exps:
        ran_exps += [base_dir]
        os.system(command)
        time.sleep(2)
        run_counter += 1
    else:
        print('run %s has already been launched!' % base_dir)
        collisions += 1

print('%d exps launched and %d collisions' % (runs, collisions))
