
import os
import cv2
import torch
import natsort
import numpy as np
from PIL import Image
from tqdm import tqdm
from ColorSpace import color_torch
from torch.utils.data import Dataset
from ColorSpace import colorSpace as cp

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from Networks.GE import GlobalEnhancement
from Networks.MCNet import McNet as ColorEnhancement
from Networks.SCNet import ScNet as LocalEnhancement
from Networks.SLCformer import SLCFormer as Classification 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


class DatasetEvaluation(Dataset):
    def __init__(self, lowLightDir_Global):
        self.lowLightDir_Global  = lowLightDir_Global 
        self.image_files = natsort.natsorted(os.listdir(lowLightDir_Global))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]

        lowLightImg_Global = Image.open(os.path.join(self.lowLightDir_Global, image_file))

        transform_global = ToTensor()
        lowLightImg_Global = transform_global(lowLightImg_Global)

        return lowLightImg_Global

class RGBTargetDataset(Dataset):
    def __init__(self, lowLightDir):
        self.lowLightDir  = lowLightDir 
        self.image_files = natsort.natsorted(os.listdir(lowLightDir))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        lowLightImg = Image.open(os.path.join(self.lowLightDir, image_file))
        transform = ToTensor()
        lowLightImg = transform(lowLightImg)
        return lowLightImg  

def bgr_rgb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class HSVTargetDataset(Dataset):
    def __init__(self, valImgDir):
        self.valImgDir = valImgDir
        self.image_files = natsort.natsorted(os.listdir(valImgDir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        validationImg = cv2.imread(os.path.join(self.valImgDir, image_file))
        validationImg = bgr_rgb(validationImg)
        return validationImg
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def Enhance(imgx, imggl, device):
    rows, columns, dimension = imgx.shape
    weigh_color = 0
    weigh_luminance = 0

    HSVInput  = cp.RgbToHsv(imgx)

    #HSV Components
    hueComponent = HSVInput[:, :, 0]              
    satComponent = HSVInput[:, :, 1]
    valComponent = HSVInput[:, :, 2]

    valComponentTensor = torch.from_numpy(valComponent).float().to(device).view(1, 1, rows, columns)

    image = transform(Image.fromarray(imgx)).unsqueeze(0)

    modelClass.eval()
    with torch.no_grad():
        output = modelClass(image.to(device))
        _, predicted = torch.max(output.data, 1)
        if predicted == 1:
           valEnhancement  = modelValL(valComponentTensor)
           rows1, columns1 = valEnhancement.shape[2:4]
           valEnhComponent = valEnhancement.detach().cpu().numpy().reshape([rows1, columns1])
           HSV = np.dstack((hueComponent, satComponent, valEnhComponent))
           algorithm = cp.HsvToRgb(HSV)
           weigh_color = 0.1
           weigh_luminance = 1
        else:
           algorithm  = modelValG(imggl)
           weigh_color = 0.4
           weigh_luminance = 0.9

    return  algorithm, predicted, weigh_color, weigh_luminance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelClass = Classification().to(device)
modelClass.load_state_dict(torch.load('./Models/CLASSIFICATION.pt'))


modelValL = LocalEnhancement().to(device)
modelValL.load_state_dict(torch.load('./Models/LOCAL.pt'))
modelValL.eval()

modelValG = GlobalEnhancement().to(device)
modelValG.load_state_dict(torch.load('./Models/GLOBAL.pt'))
modelValG.eval()

modelColor = ColorEnhancement().to(device)
modelColor.load_state_dict(torch.load('./Models/COLOR.pt'))
modelColor.eval()  


rgb2hsv = color_torch.RGBtoHSV().to(device)

input_dir = r"./1_Input"
output_dir = r"./2_Output"
os.makedirs(output_dir, exist_ok=True)  

dataset_test = DatasetEvaluation(input_dir)
rgb_dataset = RGBTargetDataset(input_dir)
hsv_dataset = HSVTargetDataset(input_dir)

for i in tqdm(range(len(rgb_dataset)), desc="Enhancing images"):
    rgb_input_image = rgb_dataset[i]
    hsv_input_image = hsv_dataset[i]

    rgb_input_tensor = rgb_input_image.unsqueeze(0).to(device)
    
    img_tensor_input = dataset_test[i]

    img_tensor = img_tensor_input.to(device)

    input_tensor = torch.as_tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor = rgb2hsv(input_tensor)
    
    with torch.no_grad():  
        color_output_tensor = modelColor(rgb_input_tensor)
        luminance_output_tensor, predicted, weigh_color, weigh_luminance = Enhance(hsv_input_image, input_tensor, device)
    
    output_color = (color_output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)


    if predicted == 1:
        output_luminance = np.dstack((luminance_output_tensor[:, :, 2], luminance_output_tensor[:, :, 1], luminance_output_tensor[:, :, 0]))
        output_color = cv2.cvtColor(output_color, cv2.COLOR_RGB2BGR)
        output_color = output_color.astype(np.float32) / 255.0
        output_image = output_color*weigh_color + output_luminance*weigh_luminance
        output_image = np.clip(output_image, 0, 1)
        base_filename = os.path.basename(hsv_dataset.image_files[i]) 
        output_filename_colorLuminance = os.path.splitext(base_filename)[0] + ".png" 
        output_filepath_colorLuminance = os.path.join(output_dir,output_filename_colorLuminance)
        cv2.imwrite(output_filepath_colorLuminance, output_image*255)
    
    else:
        with torch.no_grad():
            output_color = modelColor(torch.unsqueeze(img_tensor, 0))
        output_color = output_color.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img_enhancement = luminance_output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        output_image =  output_color*weigh_color + output_img_enhancement*weigh_luminance
        output_image = np.clip(output_image, 0, 1)
        output_img_save = (output_image*255).astype(np.uint8)
        base_filename = os.path.basename(hsv_dataset.image_files[i]) 
        output_filename_colorLuminance = os.path.splitext(base_filename)[0] + ".png" 
        output_filepath_colorLuminance = os.path.join(output_dir,output_filename_colorLuminance)
        cv2.imwrite(output_filepath_colorLuminance, cv2.cvtColor(output_img_save, cv2.COLOR_RGB2BGR))


print("Images enhanced and saved to", output_dir)
