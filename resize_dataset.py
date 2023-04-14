import os
import PIL
import numpy as np
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--folder", type=str)

args = parser.parse_args()

DATASET_FOLDER = "EuroSat_generated_randomScale/eurosat/2750"
new_folder = "EuroSat_generated_randomScale_64/eurosat/2750"

folders = os.listdir(DATASET_FOLDER)
trans = T.Resize(64, antialias=False)
folder = args.folder
# for folder in folders:
# folder = "Sea"
path = os.path.join(DATASET_FOLDER, folder)
imgs = os.listdir(path)

# if folder in os.listdir(new_folder):
#     print(f"{folder} already exists")
#     continue

for img_file in tqdm(imgs):
    file = os.path.join(path, img_file)
    img = torch.tensor(np.array(PIL.Image.open(file)))
    img = img.permute(2, 0, 1)
    img = trans(img)

    if not os.path.exists(os.path.join(new_folder, folder)):
        os.makedirs(os.path.join(new_folder, folder))
    
    new_img_file = folder+"_"+img_file
    plt.imsave(os.path.join(new_folder, folder, new_img_file), np.array(img.permute(1, 2, 0)))
