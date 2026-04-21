# 提取特征

import torch, torchvision
import PIL.Image as Image
import os
from torch.nn.functional import cosine_similarity
from torchvision import transforms

DATA_ROOT = "models/mtcnn_project/output/video_output/3"

def main():
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    
    # 定义预处理变换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img1 = Image.open(f"{DATA_ROOT}/ikura/0022.jpg").convert('RGB')
    img2 = Image.open(f"{DATA_ROOT}/ikura/0023.jpg").convert('RGB')
    
    # 预处理图像
    input1 = preprocess(img1).unsqueeze(0)  # 添加batch维度
    input2 = preprocess(img2).unsqueeze(0)
    
    with torch.no_grad():
        feat1 = model(input1)
        feat2 = model(input2)
    
    # 余弦相似度
    similarity = cosine_similarity(feat1, feat2)
    print(f"Cosine similarity: {similarity.item()}")

if __name__ == "__main__":
    main()
