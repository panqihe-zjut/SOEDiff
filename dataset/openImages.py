import os
import cv2
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class openimage(Dataset):

    def __init__(self, annotations_file,  thred_size, transformFlag=True,labeltype='label'):

        self.root_path = '/storage/panqihe/datasets/openimage'
        self.image_folder = self.root_path
        # self.filterclass = ['Human head','Human hair''Human face','Human arm','Human leg','Human nose','Human hand',
        #                     'Sports equipment','Auto part','Human mouth','Human eye']

        self.filterclass = ['Wheel','Human head','Human hair','Person','Human face','Human arm','Tire',
                            'Clothing','Human leg','Human nose','Human hand','Sports equipment','Auto part',
                            'Human mouth','Mammal','Man','Human body','Human eye','Girl','Woman',
                            'Flower','Window','Fashion accessory','Plant','Glasses']
        
        self.filterclass = [s.lower() for s in self.filterclass]
        self.labeltype   = labeltype

        f = open(annotations_file)
        self.annotations = json.load(f)



        
        print(' ----- LOADING DATASETS ----- ')
        self.items        = []
        for index in range(0, len(self.annotations)): 
            imageInfo = self.annotations[index]
            if  self.checksize(imageInfo, thred=thred_size) and self.checkclass(imageInfo):
                self.items.append(imageInfo)
                
        print("len of annotations", str(len(self.annotations)))
        print("len of datasets   ", str(len(self.items)))

        self.transformFlag = transformFlag

    def transform(self, image):
        # resize image
        image = image
        image = TF.resize(image, 512, interpolation=transforms.InterpolationMode.BILINEAR)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image
    
    def mask_transform(self, image):
        # resize image
        image = image
        image = TF.resize(image, 512, interpolation=transforms.InterpolationMode.BILINEAR)

        image = TF.to_tensor(image)
        return image



    def __getitem__(self, index):
        imageInfo = self.items[index]
        image = Image.open(os.path.join(self.image_folder, imageInfo['ImageID'])).convert("RGB")
        image_resized = image.resize((512, 512), Image.BICUBIC)
        box   = [imageInfo['XMin'], imageInfo['YMin'], imageInfo['XMax'], imageInfo['YMax']]
        mask  = self.box2mask(box)

        cropBox, cropImage, cropMask = self.cropImage(box, image_resized, mask)
        cropImage = cropImage.resize((512, 512), Image.BICUBIC)
        cropMask  = cropMask.resize((512, 512), Image.BICUBIC)

        if self.labeltype=='label':
            label = imageInfo['LabelName']
        if self.labeltype=='smalllabel':
            label = 'small '+imageInfo['LabelName']
        if self.labeltype=='colorlabel':
            label = imageInfo['color']+" "+imageInfo['LabelName']
        if self.labeltype=='smallcolorlabel':
            label = 'small '+imageInfo['color']+" "+imageInfo['LabelName']

        if self.transformFlag==True:
            # return {'imageInfo': imageInfo, 'image_resized':image_resized, 'image':image, 'mask':mask, 'label':imageInfo['LabelName']}
            return {'imageInfo': imageInfo, 'image':self.transform(image_resized), 'mask':self.mask_transform(mask), 'label':label,'box':box,
                    'croped_image':self.transform(cropImage), 'croped_mask':self.mask_transform(cropMask),'croped_box':cropBox, 'croped_label':label}
        else:
            return {'imageInfo': imageInfo, 'image':image_resized, 'mask':mask, 'label':label,'box':box,
                    'croped_image':cropImage, 'croped_mask':cropMask, 'croped_box':cropBox}


    def __len__(self):
        return len(self.items)

    def checkitem(self, item):
        return item['IsOccluded']==0 and item['IsTruncated']==0 and item['IsGroupOf']==0 and item['IsInside']==0
    
    def checksize(self, item, thred):
        box = [item['XMin'], item['YMin'], item['XMax'], item['YMax']]
        size = (box[2] - box[0]) * (box[3] - box[1])
        return True if thred[0]*thred[0]<=size<=thred[1]*thred[1] else False

    def checkclass(self, imageInfo):
        if imageInfo['LabelName'].lower() in self.filterclass :
            return False
        else:
            return True


    def visualize(self, index):
        info = self.__getitem__(index)
        image_resized = info['image_resized']
        image         = info['image']
        bbox          = [info['imageInfo']['XMin'], info['imageInfo']['YMin'], 
                            info['imageInfo']['XMax'], info['imageInfo']['YMax']]
        image_resized_box = self.drawbox(image_resized, bbox)
        image_box         = self.drawbox(image, bbox)
        image_box.save('test.jpg')
        image_resized_box.save('test1.jpg')
    
    def drawbox(self, image, boxes):
        draw = ImageDraw.Draw(image)
        iw, ih = image.size
        box = [boxes[0]*iw, boxes[1]*ih, boxes[2]*iw, boxes[3]*ih]
        draw.rectangle(box, outline='red', width=2)
        return image
    


    def box2mask(self, box):
        image = Image.new('RGB', (512, 512), color='black')
        draw = ImageDraw.Draw(image)
        draw.rectangle([(box[0]*512, box[1]*512), (box[2]*512, box[3]*512)], fill='white')
        return image
    
    def cropImage(self, box, image, mask):
        x1,y1,x2,y2 =  box[0],box[1],box[2],box[3]
    
        if x2-x1>=0.5 or  y2-y1>=0.5:
            return  box, image, mask


        if 0.25<=x1<=0.75 and 0.25<=y1<=0.75 and  0.25<=x2<=0.75  and  0.25<=y2<=0.75:
            centerx = (x1+x2)/2
            centery = (y1+y2)/2

            cropscale  = 0.25
            newsize =min( min((1-centerx), cropscale), min(centerx,cropscale), min((1-centery), cropscale), min(centery, cropscale) )
            crop_x1 = centerx-newsize
            crop_y1 = centery-newsize
            crop_x2 = centerx+newsize
            crop_y2 = centery+newsize

                
        elif 0<x1<0.5 and 0<y1<0.5 and  0<x2<0.5 and 0<y2<0.5: 
            crop_x1 = 0
            crop_y1 = 0
            crop_x2 = 0.5
            newsize = 0.5
            crop_y2 = 0.5

        elif 0.5<=x1<=1 and 0<y1<0.5 and  0.5<=x2<=1 and 0<y2<0.5:
            crop_x1 = 0.5
            crop_y1 = 0
            crop_x2 = 1
            newsize = 0.5
            crop_y2 = 0.5

        elif 0<x1<0.5 and 0.5<=y1<=1 and   0<x2<0.5 and  0.5<=y2<=1:
            crop_x1 = 0
            crop_y1 = 0.5
            crop_x2 = 0.5
            crop_y2 = 1
            newsize = 0.5

        
        elif  0.5<=x1<=1 and 0.5<=y1<=1 and  0.5<=x2<=1  and  0.5<=y2<=1:
            crop_x1 = 0.5
            crop_y1 = 0.5
            crop_x2 = 1
            crop_y2 = 1
            newsize = 0.5

        else:
            return  box, image, mask

        # else:
        #     centerx = (x1+x2)/2
        #     centery = (y1+y2)/2
        #     # cropscale  = 0.25
        #     # cropscale  = random.uniform(0.125, 0.25)
        #     cropscale  = 0.25
        #     # cropscale   = random.uniform(0.25, 0.5)

        #     newsize =min( min((1-centerx), cropscale), min(centerx,cropscale), min((1-centery), cropscale), min(centery, cropscale) )
        #     crop_x1 = centerx-newsize
        #     crop_y1 = centery-newsize
        #     crop_x2 = centerx+newsize
        #     crop_y2 = centery+newsize
        #     print(crop_x1, crop_y1, crop_x2, crop_y2)

        newImage = image.crop((crop_x1*512, crop_y1*512, crop_x2*512, crop_y2*512))
        newMask  = mask.crop((crop_x1*512, crop_y1*512, crop_x2*512, crop_y2*512))
        newbox   = [(x1-crop_x1)/(2*newsize), (y1-crop_y1)/(2*newsize), (x2-crop_x1)/(2*newsize),  (y2-crop_y1)/(2*newsize)]

        return newbox, newImage, newMask



class sogopenimagedataloader(
   
):
    def __init__(self,
        thred_size,
        annotations_file ,
        batch_size=12, 
        shuffle=False,
        num_workers=2,
        ):

        dataset = openimage(annotations_file=annotations_file, thred_size=thred_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.num_batches = len(dataset)/batch_size
        self.dataloader = dataloader


if __name__=='__main__':
    dataset = openimage(annotations_file='dataset/annotations-filter_openImage_training.json')
    import pdb
    pdb.set_trace()

    