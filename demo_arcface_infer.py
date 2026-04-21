from models import ArcFace
import torch
import os


@torch.no_grad()
def compare_faces(model, img1_path, img2_path, device="cpu"):
    feat1 = model.extract_feature(img1_path)
    feat2 = model.extract_feature(img2_path)

    sim = torch.sum(feat1 * feat2, dim=1)

    return sim.item()


def main():
    # 你的数据集
    DATA_ROOT = "data/yoasobi_TFT"
    img_path = f"{DATA_ROOT}/ikura/0022.jpg"
    img2_path = f"{DATA_ROOT}/ikura/0023.jpg"

    model = ArcFace(weight_path="./models/weights/arcface.pt", device="cpu")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到图片文件: {img_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    feat = model.extract_feature(img_path)
    print("feature shape =", feat.shape)
    print("feature[0][:10] =", feat[0, :10])

    if img2_path is not None:
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"找不到第二张图片: {img2_path}")
        sim = compare_faces(model, img_path, img2_path, device=device)
        print("cosine similarity =", sim)


if __name__ == "__main__":
    main()
