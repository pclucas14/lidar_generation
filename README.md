# lidar_generation

Official Pytorch implementation to generate LiDAR scans using GANs and VAEs


| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/11.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/174.png">|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/548.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/626.png">|
|<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/859.png"> |<img width="1604" src="https://github.com/pclucas14/lidar_generation/blob/master/samples/887.png">|

(samples from the proposed GAN model)

###  running code

#### GAN experiments
`python gan_2d.py <flags>`

#### VAE experiments
`python vae_2d.py <flags>`

### trained models
Pretrained model weights are available in `trained_models` directory. Use the function `load_model_from_file` to properly load the weights. 

## Aknowledgements
Thanks to [Fxia2] (https://github.com/fxia22/) for his NNDistance module. 
Thanks to [Thibault GROUEIX] (https://github.com/ThibaultGROUEIX) and [Panos Achlioptas] (https://github.com/optas) for open sourcing their code. 
