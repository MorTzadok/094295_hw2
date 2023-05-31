import copy
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as T
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from PIL import Image, ImageEnhance

base_dir = os.path.join("..", "data copy")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

#   MOVE FILES 
#movung 20 percent of the trainn data to val data
for number in  os.listdir(train_dir) :
    files = os.listdir(train_dir + '/' + str(number))
    no_of_files = len(files) // 3
    for file_name in random.sample(files, no_of_files):
        os.rename(train_dir + '/' + str(number) + '/' + file_name, val_dir + '/' + str(number) + '/' + file_name)

# CONCATINATE IMAGES

# v_files = [file for file in os.listdir(f"../data copy/concat/v") if str(file) != '.DS_Store']
# ii_files = [file for file in os.listdir(f"../data copy/concat/ii") if str(file) != '.DS_Store']
# iii_files = [file for file in os.listdir(f"../data copy/concat/iii") if str(file) != '.DS_Store']

# for i,v_im in enumerate(v_files): 
#     for j,ii_im in enumerate(ii_files):
#         # load iages
#         v = Image.open("../data copy/concat/v/" + str(v_im))
#         w1, h1 = v.size

#         ii = Image.open("../data copy/concat/ii/" +str(ii_im))
#         w2, h2 = ii.size
#         # combine images
#         vii = Image.new('L',(w1 + w2,h1))
#         vii.paste(v)
#         vii.paste(ii, box=(w1, 0))
#         vii = vii.resize((round(v.size[0]), round(v.size[1])))
#         #saving the new image
#         dest_path = f"../data copy/train/vii"
#         vii.save(dest_path+'/'+str(i)+'concat'+ str(j)+".png")


# for i,v_im in enumerate(v_files): 
#     for j,iii_im in enumerate(iii_files):
#         # load iages
#         v = Image.open("../data copy/concat/v/" + str(v_im))
#         w1, h1 = v.size

#         iii = Image.open("../data copy/concat/iii/" +str(iii_im))
#         w2, h2 = iii.size
#         # combine images
#         viii = Image.new('L',(w1 + w2,h1))
#         viii.paste(v)
#         viii.paste(iii, box=(w1, 0))
#         viii = viii.resize((round(v.size[0]), round(v.size[1])))
#         #saving the new image
#         dest_path = f"../data copy/train/viii"
#         viii.save(dest_path+'/'+str(i)+'concat' +str(j)+".png")

new_images = {'i':[],'ii':[],'iii':[],'iv':[],'v':[],'vi':[],'vii':[],'viii':[],'ix':[],'x':[]}
for number in os.listdir(train_dir):
    number_path = f"{train_dir}/{number}"
#  I, II, III, IV, V, VI, VII, VIII, IX, and X
    number_files = [file for file in os.listdir(number_path) if str(file) != '.DS_Store']
    #     VERTICAL FLIP
    #if the number doesnt contain v we can flip up the image without changing the number
    if str(number)  in ['i','ii','iii','ix','x']:
        for image_path in number_files: 
            image = Image.open(number_path+'/'+image_path)
            modified_img = image.transpose(Image.FLIP_TOP_BOTTOM)
            new_images[number].append(modified_img)
            
    
    #     HORIZONTAL FLIP
    #if the number doesnt contain 2 types of digits we can flip horizontaly the image without changing the number
    if str(number) in ['i','ii','iii','v','x']:
        for image_path in number_files: 
            image = Image.open(number_path+'/'+image_path)
            modified_img = image.transpose(Image.FLIP_LEFT_RIGHT)
            new_images[number].append(modified_img)

    # ROTATE 180
    if str(number)  in ['i','ii','iii','x']:
        for image_path in number_files: 
            image = Image.open(number_path+'/'+image_path)
            modified_img = image.transpose(Image.FLIP_TOP_BOTTOM)
            new_images[number].append(modified_img)

    #   ROTATE 90 270 
    #we can rotate the images of x by 90 degrees to the right or to the left  without changing the number
    if str(number) not in ['iv','vi']:
        for i,image_path in enumerate(number_files): 
            if  i%3 != 0:
                image = Image.open(number_path+'/'+image_path)
                vertical_img = image.rotate(90)
                new_images[number].append(modified_img) 
                modified_img = image.rotate(270)
                new_images[number].append(modified_img)

    #   FLIP & CHANGE NUMBER - VI <--> IV
    # if we flip to the right some numbers we get different numbers
    if str(number) in ['iv','vi']:
        for image_path in number_files: 
            image = Image.open(number_path+'/'+image_path)
            modified_img = image.transpose(Image.FLIP_LEFT_RIGHT)
            if str(number) == 'vi':
                new_images['iv'].append(modified_img)
            else:
                new_images['vi'].append(modified_img)
   
    #   SAVE
    # after adding all modified images in new_images dictionery, lets save them as new_data file
for folder_name, images in new_images.items():
    dest_path = f"../data copy/train/{folder_name}"
    # print("saving images to " + str(folder_name))
    print(str(folder_name)+ " - num of  additional images : " + str(len(images)))
    for i,image in enumerate(images):
        # os.rename(f"some_path2/{image}", f"{dest_path}/{image}")
        image.save(dest_path+'/'+str(i)+".png")
    # print("finish saving images to " + str(folder_name))


#  ADD NOISE
for number in os.listdir(train_dir) :
    number_path = f"{train_dir}/{number}"
    number_files = [file for file in os.listdir(number_path) if str(file) != '.DS_Store']

    for i,image_path in enumerate(number_files): 
        if  i%14 != 0:
            image = Image.open(number_path+'/'+image_path)
            modified_image = image.effect_spread(10)
            dest_path = f"../data copy/train/{number}"
            modified_image.save(dest_path+'/n'+str(i)+".png")

            # modifie brightness
        # if  i%4 == 0 and i%6 !=0:
        #     enhancer = ImageEnhance.Brightness(image)
        #     factor =  random.uniform(-1, 1)
        #     modified_image = enhancer.enhance(factor)
        #     dest_path = f"../data copy/train/{number}"
        #     modified_image.save(dest_path+'/n'+str(i)+".png")

## AUGMENTATION ON VAL 


# new_images = {'i':[],'ii':[],'iii':[],'iv':[],'v':[],'vi':[],'vii':[],'viii':[],'ix':[],'x':[]}
# for number in os.listdir(val_dir):
#     number_path = f"{val_dir}/{number}"
# #  I, II, III, IV, V, VI, VII, VIII, IX, and X
#     number_files = [file for file in os.listdir(number_path) if str(file) != '.DS_Store']
#     #     VERTICAL FLIP
#     #if the number doesnt contain v we can flip up the image without changing the number
#     if str(number)  in ['i','ii','iii','ix','x']:
#         for image_path in number_files: 
#             image = Image.open(number_path+'/'+image_path)
#             modified_img = image.transpose(Image.FLIP_TOP_BOTTOM)
#             new_images[number].append(modified_img)
            


#     #   FLIP & CHANGE NUMBER - VI <--> IV
#     # if we flip to the right some numbers we get different numbers
#     if str(number) in ['iv','vi']:
#         for image_path in number_files: 
#             image = Image.open(number_path+'/'+image_path)
#             modified_img = image.transpose(Image.FLIP_LEFT_RIGHT)
#             if str(number) == 'vi':
#                 new_images['iv'].append(modified_img)
#             else:
#                 new_images['vi'].append(modified_img)
   
#     #   SAVE
#     # after adding all modified images in new_images dictionery, lets save them as new_data file
# for folder_name, images in new_images.items():
#     dest_path = f"../data copy/val/{folder_name}"
#     # print("saving images to " + str(folder_name))
#     print(str(folder_name)+ " - num of  additional images : " + str(len(images)))
#     for i,image in enumerate(images):
#         # os.rename(f"some_path2/{image}", f"{dest_path}/{image}")
#         image.save(dest_path+'/'+str(i)+".png")
#     # print("finish saving images to " + str(folder_name))


# #  ADD NOISE
# for number in os.listdir(train_dir):
#     number_path = f"{val_dir}/{number}"
#     number_files = [file for file in os.listdir(number_path) if str(file) != '.DS_Store']

#     for i,image_path in enumerate(number_files): 
#         if  i%4 == 0:
#             image = Image.open(number_path+'/'+image_path)
#             modified_image = image.effect_spread(10)
#             dest_path = f"../data copy/val/{number}"
#             modified_image.save(dest_path+'/n'+str(i)+".png")