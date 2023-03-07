from tqdm import tqdm
import numpy as np
import torch


def train(data_loader, model, loss_fn, loss_fn2, optimizer, epoch, writer, config):
    model.train()
    loss_crossEntropy = []
    total = 0
    iter_dict = {
        "actual": {}, "face": {}, "ocu": {},
    }

    for i, (input1, input2, target1, target2, target3) in enumerate(tqdm(data_loader)):
        # each pred consists of [subject, ethnic, gender]
        face_fea, ocu_fea, face_pred, ocu_pred = model.forward(input1.to(config.device), input2.to(config.device),
                                                               return_feature=True)
        target_list = [target1.to(config.device), target2.to(config.device), target3.to(config.device)]
        loss_t = []

        for j, tgt in enumerate(target_list):
            cm_loss = loss_fn2[j](face_fea[:, j, :], ocu_fea[:, j, :])
            lm_loss = loss_fn[j](s1=face_pred[j], s2=ocu_pred[j], target=tgt)
            loss_t.append(lm_loss + cm_loss)
        total_l = torch.sum(torch.stack(loss_t))
        loss_crossEntropy.append(total_l)
        total += target1.size(0)

        optimizer.zero_grad()
        total_l.backward()
        optimizer.step()

        for k, pred_type in enumerate(["subject", "ethnic", "gender"]):
            if i == 0:  # first iteration
                iter_dict["actual"][pred_type] = target_list[k]
                iter_dict["face"][pred_type] = torch.argmax(face_pred[k], dim=1)
                iter_dict["ocu"][pred_type] = torch.argmax(ocu_pred[k], dim=1)
            else:
                iter_dict["actual"][pred_type] = torch.cat((iter_dict["actual"][pred_type], target_list[k]))
                iter_dict["face"][pred_type] = torch.cat(
                    (iter_dict["face"][pred_type], torch.argmax(face_pred[k], dim=1)))
                iter_dict["ocu"][pred_type] = torch.cat((iter_dict["ocu"][pred_type], torch.argmax(ocu_pred[k], dim=1)))

    train_acc_dict = {
        "face": {}, "ocu": {}
    }
    for pred_type in ["subject", "ethnic", "gender"]:
        train_acc_dict["face"][pred_type] = torch.sum(
            iter_dict["face"][pred_type] == iter_dict["actual"][pred_type]).item() / total
        train_acc_dict["ocu"][pred_type] = torch.sum(
            iter_dict["ocu"][pred_type] == iter_dict["actual"][pred_type]).item() / total
    for key in train_acc_dict:
        writer.add_scalar("train/{}/subj_acc".format(key), train_acc_dict[key]["subject"], epoch)
        writer.add_scalar("train/{}/ethn_acc".format(key), train_acc_dict[key]["ethnic"], epoch)
        writer.add_scalar("train/{}/gend_acc".format(key), train_acc_dict[key]["gender"], epoch)

    total_loss = torch.mean(torch.tensor(loss_crossEntropy))
    writer.add_scalar("train/total_loss", total_loss, epoch)

    return total_loss, train_acc_dict


def valid(data_loader, model, loss_fn, loss_fn2, epoch, writer, config):
    model.eval()
    loss_crossEntropy = []
    total = 0
    iter_dict = {
        "actual": {}, "face": {}, "ocu": {}
    }

    for i, (input1, input2, target1, target2, target3) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            # each pred consists of [subject, ethnic, gender]
            face_fea, ocu_fea, face_pred, ocu_pred = model.forward(input1.to(config.device), input2.to(config.device),
                                                                   return_feature=True)

        target_list = [target1.to(config.device), target2.to(config.device), target3.to(config.device)]
        loss_t = []

        for j, tgt in enumerate(target_list):
            cm_loss = loss_fn2[j](face_fea[:, j, :], ocu_fea[:, j, :])
            lm_loss = loss_fn[j](s1=face_pred[j], s2=ocu_pred[j], target=tgt)
            loss_t.append(lm_loss + cm_loss)
        total_l = torch.sum(torch.stack(loss_t))
        loss_crossEntropy.append(total_l)
        total += target1.size(0)

        for k, pred_type in enumerate(["subject", "ethnic", "gender"]):
            if i == 0:  # first iteration
                iter_dict["actual"][pred_type] = target_list[k]
                iter_dict["face"][pred_type] = torch.argmax(face_pred[k], dim=1)
                iter_dict["ocu"][pred_type] = torch.argmax(ocu_pred[k], dim=1)
            else:
                iter_dict["actual"][pred_type] = torch.cat((iter_dict["actual"][pred_type], target_list[k]))
                iter_dict["face"][pred_type] = torch.cat(
                    (iter_dict["face"][pred_type], torch.argmax(face_pred[k], dim=1)))
                iter_dict["ocu"][pred_type] = torch.cat((iter_dict["ocu"][pred_type], torch.argmax(ocu_pred[k], dim=1)))

    val_acc_dict = {
        "face": {}, "ocu": {}
    }
    for pred_type in ["subject", "ethnic", "gender"]:
        val_acc_dict["face"][pred_type] = torch.sum(
            iter_dict["face"][pred_type] == iter_dict["actual"][pred_type]).item() / total
        val_acc_dict["ocu"][pred_type] = torch.sum(
            iter_dict["ocu"][pred_type] == iter_dict["actual"][pred_type]).item() / total
    for key in val_acc_dict:
        writer.add_scalar("valid/{}/subj_acc".format(key), val_acc_dict[key]["subject"], epoch)
        writer.add_scalar("valid/{}/ethn_acc".format(key), val_acc_dict[key]["ethnic"], epoch)
        writer.add_scalar("valid/{}/gend_acc".format(key), val_acc_dict[key]["gender"], epoch)

    total_loss = torch.mean(torch.tensor(loss_crossEntropy))

    writer.add_scalar("valid/total_loss", total_loss, epoch)

    return total_loss, val_acc_dict


def evaluate_crossmodal_data_features_dict(base_data, test_data, base_gt, test_gt, method="max"):
    total_true_preds = 0
    cos_sim_score_list = []
    for i, test_data_feature in enumerate(test_data):
        cos_sim_score_list.append(torch.nn.functional.cosine_similarity(
            test_data_feature.repeat(len(base_data), 1), base_data))

    for i, cos_sim_score in enumerate(tqdm(cos_sim_score_list)):
        if method == "max":
            if base_gt[torch.argmax(cos_sim_score).item()] == test_gt[i]:
                total_true_preds += 1
        else:
            raise "{} is not implemented yet".format(method)

    return total_true_preds / len(test_data)
