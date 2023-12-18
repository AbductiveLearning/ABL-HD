import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.resnet import resnet34

scale_size = 96
normalize = transforms.Normalize(mean=[0.713], std=[0.225])
test_data_transform = transforms.Compose(
    [transforms.ConvertImageDtype(torch.float), normalize]  # transforms.ToTensor(),
)


class ResNet34(nn.Module):
    def __init__(self, num_class, pretrained_path=None):
        super(ResNet34, self).__init__()

        self.net = resnet34(num_classes=num_class)
        self.net.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path), strict=True)

    def forward(self, x):
        out = self.net(x)
        return out

    def predict_imgs_classes(self, imgs):
        data = torch.stack(imgs)
        data = data.cuda()
        data = test_data_transform(data)
        with torch.inference_mode():
            out = self(data.half())
            out = out.argmax(axis=1).cpu()
        return out

    def predict_classes(self, data_loader):
        results_out = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda(non_blocking=True)
                out = self(data).argmax(axis=1).cpu()
                results_out.append(out)
        return torch.cat(results_out, axis=0)


class TorchDataset(Dataset):
    def __init__(self, img_list, scale_size=96):
        self.img_list = img_list
        normalize = transforms.Normalize(mean=[0.713], std=[0.225])
        self.test_data_transform = transforms.Compose(
            [
                transforms.Resize((scale_size, scale_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = self.test_data_transform(img)
        label = 0
        return img, label
