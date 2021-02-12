'''
		CamSeq01 Dataset,
  Cambridge-Toyota Labeled Objects in Video, 

Download zip file (90 Mb)
CamSeq01 Dataset
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip


 CamSeq01 is a groundtruth dataset that can be freely used only for research work in object recognition in video. 

If you intend to use this database, please cite the following paper:
   Julien Fauqueur, Gabriel Brostow, Roberto Cipolla, 
   "Assisted Video Object Labeling By Joint Tracking of Regions and Keypoints", 
   IEEE International Conference on Computer Vision (ICCV'2007) 
   Interactive Computer Vision Workshop. Rio de Janeiro, Brazil, October 2007

For more information, please visit: 
http://www.eng.cam.ac.uk/~jf330/CamSeq01/

    Label Colors:
                    64 128 64	Animal
                    192 0 128	Archway
                    0 128 192	Bicyclist
                    0 128 64	Bridge
                    128 0 0		Building
                    64 0 128	Car
                    64 0 192	CartLuggagePram
                    192 128 64	Child
                    192 192 128	Column_Pole
                    64 64 128	Fence
                    128 0 192	LaneMkgsDriv
                    192 0 64	LaneMkgsNonDriv
                    128 128 64	Misc_Text
                    192 0 192	MotorcycleScooter
                    128 64 64	OtherMoving
                    64 192 128	ParkingBlock
                    64 64 0		Pedestrian
                    128 64 128	Road
                    128 128 192	RoadShoulder
                    0 0 192		Sidewalk
                    192 128 128	SignSymbol
                    128 128 128	Sky
                    64 128 192	SUVPickupTruck
                    0 0 64		TrafficCone
                    0 64 64		TrafficLight
                    192 64 128	Train
                    128 128 0	Tree
                    192 128 192	Truck_Bus
                    64 0 64		Tunnel
                    192 192 0	VegetationMisc
                    0 0 0		Void
                    64 192 0	Wall

'''

import torch
import torch.nn as nn
from torch.utils import data
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import wget
import shutil
import zipfile

class CamSeqDataset(data.Dataset):

    # These are the colors given to each class [0..31] (32 classes)
    segmentation_classes_colors = np.array([[64, 128, 64],[192, 0, 128],[0, 128, 192],[0, 128, 64],[128, 0, 0],[64, 0, 128],[64, 0, 192],[192, 128, 64],[192, 192, 128],[64, 64, 128],[128, 0, 192],[192, 0, 64],[128, 128, 64],[192, 0, 192],[128, 64, 64],[64, 192, 128],[64, 64, 0],	[128, 64, 128],[128, 128, 192],[0, 0, 192],	[192, 128, 128],[128, 128, 128],[64, 128, 192],[0, 0, 64],[0, 64, 64],[192, 64, 128],[128, 128, 0],[192, 128, 192],[64, 0, 64],[192, 192, 0],[0, 0, 0],[64, 192, 0]])
    
    # Which device our tensors should use (cpu/cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        super(CamSeqDataset, self).__init__()

        # Download the dataset if it isn't 
        CamSeqDataset.download()
        # Unzip and structure images/masks
        CamSeqDataset.unzip()
        folder_path = "CamSeq"
        
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.png'))

        self.imgs = []
        self.mask_files = []
        self.masks = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path.replace(".png","_L.png")))) 
            
            image = cv2.imread(img_path)
            label = cv2.imread(self.mask_files[-1])
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_AREA)
            n_classes = 32
            one_hot_label = np.zeros((256, 256, n_classes))
            for i, class_color in enumerate(CamSeqDataset.segmentation_classes_colors):

                # Close colors pixels, no need for 3 channel dim since it's either 3 True, or 3 False 
                # and we need (256, 256) bool array for slicing one_hot 
                class_pixels = np.isclose(label[:,:,:3], class_color, rtol=0, atol=20)[:,:,0]
                one_hot_label[:, :, i][class_pixels] = 1
            
            self.imgs.append(image)
            self.masks.append(one_hot_label)

    def __getitem__(self, index):

        image = self.imgs[index]
        label = self.masks[index]
        
        image = torch.from_numpy(image).float().permute(2,0,1).to(CamSeqDataset.device)
        label = torch.from_numpy(label).float().permute(2,0,1).to(CamSeqDataset.device)
        return image, label 

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def download():
        
        # If the zip file isn't downloaded
        if "CamSeq.zip" not in glob.glob("*"):

            wget.download(
                "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip",
                out="CamSeq.zip"
            )
        else:
            print("** CamSeq Dataset Zip file exists **")
    
    @staticmethod
    def unzip():
        
        if "CamSeq" in glob.glob("*"):            
            print("** CamSeq Dataset folder exists **")
        else:
            os.mkdir("CamSeq")
            with zipfile.ZipFile("CamSeq.zip", "r") as zipped_file:
                zipped_file.extractall("CamSeq/")

            os.mkdir("CamSeq/images")
            os.mkdir("CamSeq/masks")

            for filename in glob.glob("CamSeq/*_L.png"):
                shutil.move(filename, "CamSeq/masks")
            
            for filename in glob.glob("CamSeq/*.png"):
                shutil.move(filename, "CamSeq/images")
            

        
class Util():


    '''
        given a tensor labeled_image (32, 256, 265)
        it construct a numpy array rgb image with respective class colors
        
        return segmented image : numpy array (256, 256, 3)
    '''
    @staticmethod
    def semantic_to_rgb(labeled_image):

        max_idx = torch.argmax(labeled_image, 0, keepdim=True)
        max_idx.shape
        rgb_img = np.zeros((256, 256, 3))

        for i, class_color in enumerate(CamSeqDataset.segmentation_classes_colors):

            class_pixels = torch.eq(max_idx, torch.tensor([i]).to(CamSeqDataset.device))
            rgb_img[:, :, :3][class_pixels.squeeze(0).cpu().detach().numpy()] = class_color
        return rgb_img

    
    @staticmethod
    def plot_prediction_example(model, dataset):

        image, ture_label = dataset[torch.randint(0, 101, (1,))[0]]

        model.eval()

        label = model(image.unsqueeze(0))
        label = nn.functional.softmax(label, dim=1).squeeze(0)

        rgb_img = Util.semantic_to_rgb(label)

        np_image = image.permute(1,2,0).cpu().detach().numpy()
        plt.subplot(1, 3, 1)
        plt.imshow(np_image.astype(np.uint8))

        plt.subplot(1, 3, 2).set_title("Pred")

        plt.imshow((rgb_img).astype(np.uint8))

        plt.subplot(1, 3, 3).set_title("Truth")

        rgb_img = Util.semantic_to_rgb(ture_label)

        plt.imshow((rgb_img).astype(np.uint8))
        plt.show()