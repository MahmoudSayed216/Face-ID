import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from models import get_model
import cv2
import numpy as np



@torch.no_grad()
def inference(model, img):
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (112, 112))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    # net = get_model(name, fp16=False)
    vec = model(img).numpy()
    vec = vec.squeeze()
    # print(feat)
    print("NUMBER OF PARAMETRS: ", sum(p.numel() for p in model.parameters()))
    return vec



# weights = torch.load(WEIGHT_PATH, map_location=torch.device("cpu"))
# model = get_model("r100", fp16 = True)
# model.load_state_dict(weights)
# model.eval()

# vec = inference(model, IMAGE_PATH)

# print(vec)
# print(vec.shape)
# WEIGHT_PATH = "../models/FaceEmbedding/ms1mv3_arcface_r100_fp16.pth"
# IMAGE_PATH = "../input/IMG_0473.jpg"

def embed(model_name: str, weights_path: str, fp16:bool, tensor: np.ndarray):
    weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = get_model(model_name)
    model.load_state_dict(weights)
    model.eval()
    vec = inference(model, tensor)

    return vec
