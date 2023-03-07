from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

from models.HA_ViT import HA_ViT
from utils.CMFP_dataset import CMFP_dataset
from utils.loss import total_LargeMargin_CrossEntropy, CrossCLR_Modality_loss
from utils.model_utils import *
import config as config


def parse_arguments(argv):
    """
        Parameters for calling eval.py
        e.g., python eval.py --dataset_mode "ShapeNet"

    :param argv: --dataset_mode {"ShapeNet", "Ours"}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_mode', help='Dataset for evaluation. {CMFP_dataset}', default='CMFP_dataset')

    return parser.parse_args(argv)


def training():
    cur_acc_dict = {
        "face": {"subject": 0., "ethnic": 0., "gender": 0.},
        "ocu": {"subject": 0., "ethnic": 0., "gender": 0.}
    }
    writer = SummaryWriter(log_dir=config.save_dir + '/summary')
    os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)

    # Dataset
    train_dataset = CMFP_dataset(config, dataset_mode="train", train_augmentation=True, imagesize=config.image_size)
    valid_dataset = CMFP_dataset(config, dataset_mode="valid", train_augmentation=False, imagesize=config.image_size)

    # Model
    model = HA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                   embed_dim=config.embed_dim,
                   num_classes_list=(config.num_sub_classes, config.num_enth_classes, config.num_gender_classes),
                   layer_depth=config.layer_depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                   norm_layer=None, drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                   drop_path_rate=config.drop_path_rate, network_type=config.network_type,
                   reuse_classifier=config.reuse_classifier).to(config.device)
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)

    loss_subj = total_LargeMargin_CrossEntropy().to(config.device)
    loss_ethn = total_LargeMargin_CrossEntropy().to(config.device)
    loss_gend = total_LargeMargin_CrossEntropy().to(config.device)

    loss_cm_subj = CrossCLR_Modality_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                                          config=config).to(config.device)
    loss_cm_ethn = CrossCLR_Modality_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                                          config=config).to(config.device)
    loss_cm_gend = CrossCLR_Modality_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                                          config=config).to(config.device)

    if config.pretrained_weights is not None:
        print("Loading Pretrained Model")
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device), strict=True)

    # Training
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    for epoch in range(config.start_epoch, config.total_epochs):
        print("[EPOCH {}/{}]".format(epoch, config.total_epochs - 1))
        train_loss, train_acc_dict = train(data_loader=train_data_loader, model=model,
                                           loss_fn=[loss_subj, loss_ethn, loss_gend],
                                           loss_fn2=[loss_cm_subj, loss_cm_ethn, loss_cm_gend], optimizer=optimizer,
                                           epoch=epoch, writer=writer, config=config)
        valid_loss, valid_acc_dict = valid(data_loader=valid_data_loader, model=model,
                                           loss_fn=[loss_subj, loss_ethn, loss_gend],
                                           loss_fn2=[loss_cm_subj, loss_cm_ethn, loss_cm_gend],
                                           epoch=epoch, writer=writer, config=config)

        if valid_acc_dict["ocu"]["subject"] >= cur_acc_dict["ocu"]["subject"]:
            cur_acc_dict = valid_acc_dict
            torch.save(model.state_dict(), "{}/checkpoints/best_model.pt".format(config.save_dir))
        torch.save(model.state_dict(), "{}/checkpoints/latest_model.pt".format(config.save_dir))

        print("    [Train] loss: {:.4f}, \nacc: {}\n".format(train_loss, train_acc_dict))
        print("    [Valid] loss: {:.4f},\nacc: {}\n".format(valid_loss, valid_acc_dict))
        print("    [BEST] acc: {}".format(cur_acc_dict))
