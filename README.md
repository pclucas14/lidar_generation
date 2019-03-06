# Deep Generative Modeling of LiDAR data

Code for reproducing all results in our paper, which can be found [here](https://arxiv.org/abs/1812.01180) </br>
Additional results can be found [here](https://github.com/pclucas14/lidar_generation/#Additional-Results)

## (key) Requirements 
- Pytorch 0.4.1/0.4.0
- TensorboardX
- Mayavi (for display only)

## Structure

    ├── Evaluation 
        ├── eval.py             # Evaluation of conditional generation, using Chamfer and EMD metrics
        ├── generate.py         # Generate point clouds from a pretrained model, both for clean and corrupted input        
    ├── Launch Scripts          
        ├── vae_rs.py           # launch the hyperparameter search used for the proposed VAE models
        ├── baseline_rs.py      # launch the hyperparameter search used for the baseline models
    ├── Trained Models  
        ├── ....                # contains the weights of the models (ours and baseline) used in the paper
    ├── Training Logs
        ├── ....                # Contains all the tensorboard logs for all the model hyperparameter searches
    ├── gan_2d.py               # Training file for GAN model
    ├── vae_2d.py               # Training file for the VAE models
    ├── models.py               # Implementation of all the model architectures used in the paper
    ├── utils.py                # Utilies for preprocessing and visualitation 

Models are logged according to the following file sturcture, where the root is specified using the `--base_dir` flag

    ├── <base_dir>
        ├── TB/
            ├── ...             # Contains the Tensorboard logs 
        ├── models/  
            ├── ...             # Contains the saved model weights, stored as `.pth` files.
    ├── args.json               # List of all the hyperparameters used for training.     


    
## Train a model
We provide the full list of commands to replicate all of our results. 

### VAE experiments
The general command is 
`python vae_2d.py <flags>` e.g. <br/>
`python vae_2d.py --z_dim=256 --batch_size=64 --kl_warmup_epochs=100` <br/> <br/>
To get more information regarding the different flags, you can run `python vae_2d.py --help` <br/>
To replicate the results from a specific model, simply provide the hyperparemeter values listed in `trained_models/<model_you_want>/args.json`

### GAN experiments
Similarly, you can train a GAN using the following command
`python gan_2d.py <flags>` e.g. <br/>
`python gan_2d.py --optim=Adam --batch_size=64 --loss=1` <br/> <br/>
To get more information regarding the different flags, you can run `python gan_2d.py --help` <br/>

## Evaluate a model
To evaluate a AE/VAE on the clean, noisy and corrupted tasks, run <br/>
`python evaluation/eval.py <path_to_base_dir_of_trained_model> <epoch_#_to_load> <emd/chamfer>` e.g. <br/>
`python evaluation/eval.py trained_models/ae_xyz 209 chamfer`, <br/> which will print the reconstruction results and the noise std /missing data percentage

## Examine training logs
Additionally, we provide the tensorboard logs for all models trained in the hyperparameter search. To see them in tensorboard, run <br/>
`tensorboard --logdir=trained_models/`

## Additional Results

#### reconstructions
Our model learns a compressed representation of the input. Here we encode into a 512 dimensional vector, (a compression of 98%) and decode it back.

| Original Lidar | Reconstruction from compressed (98%) encoding |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/real_gif.gif"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/polar_clean.gif">|


Surprisingly, our approch is highly robust to noisy input. Here we added gaussian noise on the input (shown on the left) rendering the lidar uninformative to the human eye. Yet, the model is able to reconstruct the point cloud with little performance loss. We note that the model  **was not trained on noisy data :O**

| Noisy input | Reconstruction from noisy input|
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/corrupt_0.3.gif"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/xyz_0.3.gif">|

Here we repeat the same corruption process, but with even more noise

| Noisy input | Reconstruction from noisy input|
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/corrupt_0.7.gif"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/gifs/xyz_0.7.gif">|

#### samples from GAN model
| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/11.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/174.png">|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/548.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/626.png">|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/859.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/887.png">|



## Aknowledgements
Thanks to [Fxia2](https://github.com/fxia22/) for his NNDistance module. 
Thanks to [Thibault GROUEIX](https://github.com/ThibaultGROUEIX) and [Panos Achlioptas](https://github.com/optas) for open sourcing their code. </br>
Thanks to [Alexia JM](https://github.com/AlexiaJM/RelativisticGAN) for her open source code on Relativistic GANs. Please check out her work if you are working with GANs! <br/>
Thanks to Alexandre Bachaalani for his video editing help!

