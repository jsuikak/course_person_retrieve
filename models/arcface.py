import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2


class IRBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if downsample or in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class IR50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)

        self.layer1 = nn.Sequential(
            IRBlock(64, 64, stride=2, downsample=True),
            IRBlock(64, 64),
            IRBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            IRBlock(64, 128, stride=2, downsample=True),
            IRBlock(128, 128),
            IRBlock(128, 128),
            IRBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            IRBlock(128, 256, stride=2, downsample=True),
            *[IRBlock(256, 256) for _ in range(13)]
        )
        self.layer4 = nn.Sequential(
            IRBlock(256, 512, stride=2, downsample=True),
            IRBlock(512, 512),
            IRBlock(512, 512),
        )

        self.bn2 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.features = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x

    @staticmethod
    def load_model(weight_path="", device="cpu"):
        if weight_path == "":
            raise ValueError("权重文件路径为空")
        model = IR50().to(device)

        ckpt = torch.load(weight_path, map_location=device)

        if not isinstance(ckpt, dict):
            raise TypeError(f"权重不是 dict/state_dict，实际类型是: {type(ckpt)}")

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        if msg.missing_keys:
            print("missing_keys =", msg.missing_keys)
        if msg.unexpected_keys:
            print("unexpected_keys =", msg.unexpected_keys)

        model.eval()
        return model

# ArcFace模型封装
class ArcFace:
    def __init__(self, weight_path="", device="cpu"):
        self.model = IR50.load_model(weight_path, device=device)
        self.device = device

    @staticmethod
    def preprocess_image(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"读取图片失败: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32)

        # ArcFace 常见预处理
        img = (img - 127.5) / 128.0

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img).unsqueeze(0)  # [1, 3, 112, 112]
        return img

    @torch.no_grad()
    def extract_feature(self, img_path):
        # 预处理后送进模型
        x = self.preprocess_image(img_path).to(self.device)
        feat = self.model(x)
        feat = F.normalize(feat, dim=1)
        return feat

    def __call__(self, img_path):
        return self.extract_feature(img_path)