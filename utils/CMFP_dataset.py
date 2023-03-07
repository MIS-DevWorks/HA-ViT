import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import natsort
import random
import numpy as np
from PIL import Image


class CMFP_dataset(torch.utils.data.Dataset):
    def __init__(self, config, dataset_mode, train_augmentation=False, imagesize=112, biometric_mode="multi"):
        self.image_size = imagesize
        self.biometric_mode = biometric_mode
        self.train_augmentation = train_augmentation

        if train_augmentation:
            self.transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1.0),
                                             ratio=(0.75, 1.33)),
                transforms.RandomEqualize(),
                transforms.RandomGrayscale(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.face_path = []
        self.ocular_l_path = []
        self.ocular_r_path = []
        self.subject = []
        self.gender = []
        self.ethnic = []

        ethnic_ = ['African', 'Caucasian', 'East Asian', 'Hispanic', 'Melanesian', 'Middle Eastern', 'South Asian']

        if dataset_mode == "train":
            self.data_path = config.dataset_path
            with open(self.data_path + "train_data.txt", "r") as f:
                info = [x.rstrip() for x in f]
        elif dataset_mode == "valid":
            self.data_path = config.dataset_path
            with open(self.data_path + "valid_data.txt", "r") as f:
                info = [x.rstrip() for x in f]

        for line in info:
            x = line.split(", ")
            self.face_path.append(self.data_path + ethnic_[int(x[3])] + '/' + x[0] + '/face/' + x[1])
            self.ocular_l_path.append(self.data_path + ethnic_[int(x[3])] + '/' + x[0] + '/ocular_left/' + x[1])
            self.ocular_r_path.append(self.data_path + ethnic_[int(x[3])] + '/' + x[0] + '/ocular_right/' + x[1])
            self.subject.append(int(x[2]))
            self.ethnic.append(int(x[3]))
            self.gender.append(int(x[4]))

        self.num_samples = len(self.face_path)

    def __getitem__(self, idx):
        subject = self.subject[idx]
        ethnic = self.ethnic[idx]
        gender = self.gender[idx]

        if self.train_augmentation:
            # Face images
            face_img = Image.open(self.face_path[idx]).convert('RGB')
            face_stacked_imgs = [self.transforms(face_img)]
            face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)
            # Ocular images
            ocular_l_img = Image.open(self.ocular_l_path[idx]).convert('RGB')
            ocular_r_img = Image.open(self.ocular_r_path[idx]).convert('RGB')

            ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
            ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

            return face_stacked_imgs, ocular_stacked_imgs, subject, ethnic, gender
        else:
            if self.biometric_mode == "multi" or self.biometric_mode == "multimodal":
                # Face images
                face_img = Image.open(self.face_path[idx]).convert('RGB')
                face_stacked_imgs = [self.transforms(face_img)]
                face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)
                # Ocular images
                ocular_l_img = Image.open(self.ocular_l_path[idx]).convert('RGB')
                ocular_r_img = Image.open(self.ocular_r_path[idx]).convert('RGB')

                ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
                ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

                return face_stacked_imgs, ocular_stacked_imgs, subject, ethnic, gender

            elif self.biometric_mode == "face":
                # Face images
                face_img = Image.open(self.face_path[idx]).convert('RGB')
                face_stacked_imgs = [self.transforms(face_img)]
                face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)

                return face_stacked_imgs, subject, ethnic, gender

            elif self.biometric_mode == "ocular":
                # Ocular images
                ocular_l_img = Image.open(self.ocular_l_path[idx]).convert('RGB')
                ocular_r_img = Image.open(self.ocular_r_path[idx]).convert('RGB')

                ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
                ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

                return ocular_stacked_imgs, subject, ethnic, gender

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    import config as config
    import cv2

    dataset = CMFP_dataset(config, dataset_mode="train", train_augmentation=True, biometric_mode="multimodal")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(len(data_iter)):
        face_img, ocular, subject, ethnic, gender = next(data_iter)

        face_img = (face_img[0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        ocular_l = (ocular[0, 0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        ocular_r = (ocular[0, 1] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imshow("show_img", cv2.cvtColor(np.hstack((face_img, ocular_l, ocular_r)), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
