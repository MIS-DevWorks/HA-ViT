from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import sys

from models.HA_ViT import HA_ViT
from utils.dataset import CMFP_dataset, other_dataset
from utils.loss import total_LargeMargin_CrossEntropy, CFPC_loss
from utils.model_utils import *
import config as config


def parse_arguments(argv):
    """
        Parameters for calling main.py
        e.g., python main.py --training_mode True --pretrain_mode False --dataset_mode "CMFP_dataset"

    :param argv: --training_mode True
    :param argv: --pretrain_mode False
    :param argv: --dataset_mode "CMFP_dataset"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', help='Train the mode.', default=True)
    parser.add_argument('--pretrain_mode', help='Use pretrained model weight.', default=False)
    parser.add_argument('--dataset_mode', help='Dataset for training or evaluation.', default='CMFP')

    return parser.parse_args(argv)


def training(args):
    cur_acc_dict = {
        "face": {"subject": 0., "ethnic": 0., "gender": 0.},
        "ocu": {"subject": 0., "ethnic": 0., "gender": 0.}
    }
    writer = SummaryWriter(log_dir=config.save_dir + '/summary')
    os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)

    # Dataset
    if args.dataset_mode == "CMFP":
        print("-------------------------------------------------------------------------------------")
        print("\tDataset Loading")
        train_dataset = CMFP_dataset(config, dataset_mode="train", train_augmentation=True, imagesize=config.image_size)
        valid_dataset = CMFP_dataset(config, dataset_mode="valid", train_augmentation=False, imagesize=config.image_size)
        print("-------------------------------------------------------------------------------------\n\n")
    else:
        raise ValueError("'{}' dataset is not found!".format(args.dataset_mode))

    print("-------------------------------------------------------------------------------------")
    print("\tHA-ViT Model")
    # Model
    model = HA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                   embed_dim=config.embed_dim,
                   num_classes_list=(config.num_sub_classes, config.num_enth_classes, config.num_gender_classes),
                   layer_depth=config.layer_depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                   norm_layer=None, drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                   drop_path_rate=config.drop_path_rate).to(config.device)
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)

    loss_subj = total_LargeMargin_CrossEntropy().to(config.device)
    loss_ethn = total_LargeMargin_CrossEntropy().to(config.device)
    loss_gend = total_LargeMargin_CrossEntropy().to(config.device)

    loss_cm_subj = CFPC_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                             config=config).to(config.device)
    loss_cm_ethn = CFPC_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                             config=config).to(config.device)
    loss_cm_gend = CFPC_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                             config=config).to(config.device)

    if args.pretrain_mode is True:
        print("\n\tLoading Pretrained Model: {}".format(config.pretrained_weights))
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device), strict=True)
    else:
        print("\n\tLoading Pretrained Model is set to False")
    
    # Training
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    print("-------------------------------------------------------------------------------------\n\n")

    print("-------------------------------------------------------------------------------------")
    print("\tTraining...")
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
    print("-------------------------------------------------------------------------------------\n")
        

def crossmodal_evaluation(args):
    if args.dataset_mode == "CMFP":
        pass
    elif args.dataset_mode == "other":
        print("-------------------------------------------------------------------------------------")
        print("\tDataset Loading")
        # Dataset
        base_dataset = other_dataset(config, train_augmentation=False, imagesize=config.image_size)
        test_dataset = other_dataset(config, train_augmentation=False, imagesize=config.image_size)

        base_data_loader = DataLoader(base_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        print("-------------------------------------------------------------------------------------\n\n")

        print("-------------------------------------------------------------------------------------")
        print("\tHA-ViT Model")
        # Model
        model = HA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                       embed_dim=config.embed_dim,
                       num_classes_list=(config.num_sub_classes, config.num_enth_classes, config.num_gender_classes),
                       layer_depth=config.layer_depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                       norm_layer=None, drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                       drop_path_rate=config.drop_path_rate).to(config.device)
        if config.pretrained_weights is not None:
            print("\n\tLoading Pretrained Model: {}".format(config.pretrained_weights))
            model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device), strict=True)
        print("-------------------------------------------------------------------------------------\n\n")

        print("-------------------------------------------------------------------------------------")
        print("\tIdentifying...")
        base_data_features_dict = get_features(model, base_data_loader, config)
        test_data_features_dict = get_features(model, test_data_loader, config)

        ''' base face vs test ocu '''
        face_ocu_acc_by_max = evaluate_crossmodal_data_features_dict(base_data=base_data_features_dict[0],
                                                                     test_data=test_data_features_dict[1],
                                                                     base_gt=base_data_features_dict[2],
                                                                     test_gt=test_data_features_dict[2],
                                                                     method='max')
        print("\t[TEST] Cross-identification(face to periocular) accuracy    : {:.2f}%".format(face_ocu_acc_by_max))

        ''' base ocu vs test face '''
        ocu_face_acc_by_max = evaluate_crossmodal_data_features_dict(base_data=base_data_features_dict[1],
                                                                     test_data=test_data_features_dict[0],
                                                                     base_gt=base_data_features_dict[2],
                                                                     test_gt=test_data_features_dict[2],
                                                                     method='max')
        print("\t[TEST] Cross-identification(periocular to face) accuracy    : {:.2f}%".format(ocu_face_acc_by_max))
        print("-------------------------------------------------------------------------------------\n")


def main(args):
    if args.training_mode is True:
        training(args=args)
    elif args.training_mode is False:
        crossmodal_evaluation(args=args)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
    