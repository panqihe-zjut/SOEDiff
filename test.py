import os
import shutil
from tqdm import tqdm
from utils import writeFile, prepare_pipe
from dataset.benchdata import benchdata


from utils import seed_torch
seed_torch(1111)


def benchGeneration(pipe, valdataset, savefolder, subfolder=""):
    savefolder = os.path.join(savefolder, subfolder)
    os.makedirs(os.path.join(savefolder, 'box'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'caption'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_ori'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_gen'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_mask'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_ori_crop'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'image_gen_crop'), exist_ok=True)

    for index in tqdm(range(0, len(valdataset)),  ncols=50):

        image_init, text, mask  = valdataset[index]['image'], valdataset[index]['label'], valdataset[index]['mask']
        imageInfo = valdataset[index]['imageInfo']['ImageID']
        imageid = imageInfo.split('/')[1].split('.')[0]

        box = [valdataset[index]['imageInfo']['XMin'], valdataset[index]['imageInfo']['YMin'],
            valdataset[index]['imageInfo']['XMax'], valdataset[index]['imageInfo']['YMax']]
        
        image_gen = pipe(prompt=text,image=image_init,mask_image=mask,num_inference_steps=20,).images[0]
        
        writeFile(os.path.join(savefolder, 'caption', str(index)+"_"+imageid+'.txt'), text)
        writeFile(os.path.join(savefolder, 'box', str(index)+"_"+imageid+'.txt'), 
                str(box[0])+" "+str(box[1])+" "+str(box[2])+" "+str(box[3]))
        image_init.save(os.path.join(savefolder, 'image_ori',str(index)+"_"+imageInfo.split('/')[1]))
        image_gen.save(os.path.join(savefolder, 'image_gen',str(index)+"_"+imageInfo.split('/')[1]))
        mask.save(os.path.join(savefolder, 'image_mask', str(index)+"_"+imageInfo.split('/')[1]))
        image_init.crop((box[0]*512, box[1]*512, box[2]*512, box[3]*512)).save(os.path.join(savefolder, 'image_ori_crop',str(index)+"_"+imageInfo.split('/')[1]))
        image_gen.crop( (box[0]*512, box[1]*512, box[2]*512, box[3]*512)).save(os.path.join(savefolder, 'image_gen_crop',str(index)+"_"+imageInfo.split('/')[1]))

def merge_folders(savefolder, subfolders, merged_folder_path):
    # 创建目录merged_folder_path
    merge_folder = os.path.join(savefolder, merged_folder_path)
    if not os.path.exists(merge_folder):
        os.makedirs(merge_folder)

    subfolders = [os.path.join(savefolder, i) for i in subfolders]
    
    # 遍历每个文件夹路径
    for  subfolder in subfolders:
        # 获取文件夹中的所有子目录名称
        subdirs = os.listdir(subfolder)
        
        # 遍历子目录
        for subdir in subdirs:
            # 创建目标文件夹路径
            merged_subdir_path = os.path.join(merged_folder_path, subdir)
            
            # 获取源文件夹路径
            source_subdir_path = os.path.join(subfolder, subdir)
            
            # 如果源文件夹路径存在，则合并到目标文件夹中
            if os.path.exists(source_subdir_path):
                merge_folders([source_subdir_path], merged_subdir_path)

def merge_folders(savefolder, subfolders, targetfolder):

    targetfolder = os.path.join(savefolder, targetfolder)
    if not os.path.exists(targetfolder):
        os.makedirs(targetfolder)

    subfolders = [os.path.join(savefolder, i) for i in subfolders]
    
    # 遍历每个文件夹路径
    for folder_path in subfolders:
        # 获取文件夹中的所有子目录名称
        subdirs = os.listdir(folder_path)
        
        # 遍历子目录
        for subdir in subdirs:
            # 创建目标文件夹路径
            merged_subdir_path = os.path.join(targetfolder, subdir)
            # 获取源文件夹路径
            source_subdir_path = os.path.join(folder_path, subdir)

            if os.path.exists(merged_subdir_path)==False:
                os.makedirs(merged_subdir_path, exist_ok=True)
                source_subdir_file = os.listdir(source_subdir_path)
                source_subdir_file = sorted(source_subdir_file, key=lambda x: int(x.split('_')[0]))
                source_file = [os.path.join(source_subdir_path, i) for i in source_subdir_file]
                target_file = [os.path.join(merged_subdir_path, i) for i in source_subdir_file]
                for k in range(0, len(source_file)):
                    shutil.copyfile(source_file[k], target_file[k])
            else:
                source_subdir_file = os.listdir(source_subdir_path)
                source_subdir_file = sorted(source_subdir_file, key=lambda x: int(x.split('_')[0]))
                lentarget          = len(os.listdir(merged_subdir_path))
                source_file = [os.path.join(source_subdir_path, i) for i in source_subdir_file]
                target_file = [os.path.join(merged_subdir_path, str(lentarget+int(i.split('_')[0]))+"_"+i.split('_')[1]) for i in source_subdir_file]
                for k in range(0, len(source_file)):
                    shutil.copyfile(source_file[k], target_file[k])




                    


# 2---------------------Distilling the Inpaint Model---------------------
lora_weight   = './exp/debug/checkpoint-50000/unet_lora/pytorch_lora_weights.safetensors'
vae_weight    = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
sd_weight     = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
pipe          = prepare_pipe( sd=sd_weight, vae=vae_weight, lora=lora_weight).to('cuda:0')
pipe.set_progress_bar_config(disable=True)

savefolder    = './exp_bench/lora-distill-exp7_colorlabel'
openimageval  = benchdata(dataname='openimageval' , transformFlag=False, thred_size=[1/8, 1/6],labeltype='colorlabel').dataset
cocoval       = benchdata(dataname='cocoval', transformFlag=False, thred_size=[1/8, 1/6], labeltype='colorlabel').dataset
benchGeneration(pipe, openimageval, savefolder, subfolder='openimageval')
benchGeneration(pipe, cocoval, savefolder, subfolder='cocoval')
