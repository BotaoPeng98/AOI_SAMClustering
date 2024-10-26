import faiss
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from PIL import Image

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

class chip_transforms_for_convNext2(torch.nn.Module):
    def __init__(self, light_num=4):
        super().__init__()
        if not isinstance(light_num, (int)):
            raise TypeError(f"Size should be int or sequence. Got {type(light_num)}")
        if isinstance(light_num, int) and light_num not in (3, 4):
            raise ValueError("Light_num should be 3 or 4")
        self.light_num = light_num

    def forward(self, image):
        # TODO adaptive light_num
        image = image.convert("RGB")
        # image = image.resize((1373, 849))
        # w, h = image.size
        # mid_width = int((w - 5) / 2)
        # mid_height = int((h - 5) / 2)
        # # image = Image.fromarray(image)
        # img_light1 = np.array(image.crop((0, 0, mid_width, mid_height)))
        # img_light2 = np.array(image.crop((mid_width + 5, 0, w, mid_height)))
        # img_light3 = np.array(image.crop((0, mid_height + 5, mid_width, h)))
        # img_light4 = np.array(image.crop((mid_width + 5, mid_height + 5, w, h)))
        # if self.light_num==3:
        #     new_img = np.concatenate([np.expand_dims(img_light4, axis=2), np.expand_dims(img_light3, axis=2), np.expand_dims(img_light1, axis=2)], axis=2)  # 三通道
        # elif self.light_num==4:
        #     new_img = np.concatenate([np.expand_dims(img_light4, axis=2), np.expand_dims(img_light3, axis=2),
        #                               np.expand_dims(img_light2, axis=2), np.expand_dims(img_light1, axis=2)], axis=2)  # 四通道
        # image = new_img.transpose(2, 0, 1)
        return np.array(image).transpose(2, 0, 1)

def get_transforms(dataset="CIFAR-10"):
    if (
        dataset == "CIFAR-10"
        or dataset == "CIFAR-20"
        or dataset == "STL-10"
        or dataset == "DTD"
        or dataset == "UCF101"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif (
        dataset == "ImageNet-Dogs" or dataset == "ImageNet-10" or dataset == "ImageNet"
    ):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256, interpolation=BICUBIC),
                torchvision.transforms.CenterCrop(224),
                _convert_image_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif dataset == "0305F" or dataset == "2235Y" or dataset == "05ELX" or dataset == "2235Y_cls" or dataset == "05ELX_cls" or dataset == "2235Y_light2" or dataset == "05ELX_light2":
        transforms = chip_transforms_for_convNext2(4)
    else:
        raise NotImplementedError
    return transforms


def get_dataloader(dataset="CIFAR-10", batch_size=4096, trainset_pth=None, valset_pth=None):
    transforms = get_transforms(dataset)
    assert trainset_pth is not None
    assert valset_pth is not None
    if dataset == "0305F" or dataset == "2235Y" or dataset == "05ELX" or dataset == "2235Y_cls" or dataset == "05ELX_cls" or dataset == "2235Y_light2" or dataset == "05ELX_light2":
        # data_train = ImageFolder(f"./data/{dataset}/two/train", transform=transforms)
        data_train = ImageFolder(trainset_pth, transform=transforms)
        data_test = ImageFolder(valset_pth, transform=transforms)
        # data_test = ImageFolder(f"./data/{dataset}/two/val", transform=transforms)
    else:
        raise NotImplementedError

    dataloader_train = DataLoader(
        data_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dataloader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return dataloader_train, dataloader_test


def mine_nearest_neighbors(features, topk=50):
    print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    print("Nearest neighbors computed.")
    return indices[:, 1:]


class NeighborsDataset(Dataset):
    def __init__(self, dataset_text, dataset_image, indices_text, indices_image):
        super(NeighborsDataset, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.indices_text = indices_text
        self.indices_image = indices_image
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, neighbor_text, neighbor_image
    
class NeighborsDataset_plus_convnext2(Dataset):
    def __init__(self, dataset_text, dataset_image, dataset_image_convnext2, indices_text, indices_image):
        super(NeighborsDataset_plus_convnext2, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.dataset_image_convnext2 = dataset_image_convnext2
        self.indices_text = indices_text
        self.indices_image = indices_image
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        anchor_image_convnext2 = self.dataset_image_convnext2.__getitem__(index)
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)
        neighbor_image_convnext2 = self.dataset_image_convnext2.__getitem__(neighbor_index_image)

        return anchor_text, anchor_image, anchor_image_convnext2, neighbor_text, neighbor_image, neighbor_image_convnext2
    
class NeighborsDataset_plus_convnext2_no_random(Dataset):
    def __init__(self, dataset_text, dataset_image, dataset_image_convnext2, indices_text, indices_image_convnext2):
        super(NeighborsDataset_plus_convnext2_no_random, self).__init__()

        self.dataset_text = dataset_text
        self.dataset_image = dataset_image
        self.dataset_image_convnext2 = dataset_image_convnext2
        self.indices_text = indices_text
        self.indices_image_convnext2 = indices_image_convnext2
        assert self.indices_text.shape[0] == len(self.indices_text)
        assert self.indices_image_convnext2.shape[0] == len(self.indices_image_convnext2)

    def __len__(self):
        return len(self.dataset_text)

    def __getitem__(self, index):
        anchor_text = self.dataset_text.__getitem__(index)
        anchor_image = self.dataset_image.__getitem__(index)
        anchor_image_convnext2 = self.dataset_image_convnext2.__getitem__(index)
        
        neighbor_index_text = np.random.choice(self.indices_text[index], 1)[0]
        neighbor_text = self.dataset_text.__getitem__(neighbor_index_text)
        
        neighbor_index_image_convnext2 = np.random.choice(self.indices_image_convnext2[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image_convnext2)
        neighbor_image_convnext2 = self.dataset_image_convnext2.__getitem__(neighbor_index_image_convnext2)

        return anchor_text, anchor_image, anchor_image_convnext2, neighbor_text, neighbor_image, neighbor_image_convnext2
