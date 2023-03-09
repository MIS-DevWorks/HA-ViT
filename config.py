from datetime import datetime


#   Path
dataset_path_CMFP                       = '/media/leslie/samsung/Biometrics/learning dataset/'
dataset_path_other                      = 'Sample Images/'
save_dir                                = 'experiments/{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
pretrained_weights                      = 'Pre-trained model/HA-ViT.pt'


#   Image
image_size                              = 112
in_chans                                = 3
num_sub_classes                         = 2239
num_enth_classes                        = 7
num_gender_classes                      = 2


#   Model
device                                  = "cuda:0"
patch_size                              = 8
embed_dim                               = 1024
layer_depth                             = 6
num_heads                               = 8
mlp_ratio                               = 4.
norm_layer                              = None
drop_rate                               = 0.1
attn_drop_rate                          = 0.
drop_path_rate                          = 0.


#   Training
batch_size                              = 4
init_lr                                 = 0.0001
optimizer                               = "AdamW"
momentum                                = 0.9
weight_decay                            = 0.0005
start_epoch                             = 0
total_epochs                            = 200
temperature                             = 0.02
negative_weight                         = 0.8
