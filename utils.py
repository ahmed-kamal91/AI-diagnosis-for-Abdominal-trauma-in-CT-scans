import os
import cv2
import math
from tqdm import tqdm

l = os.listdir
j = os.path.join

def get_img_pths(d_pth):

    img_paths = list()
    for p in tqdm(l(d_pth)):
        for s in l(os.path.join(d_pth, p)):

            scans = list()
            s_pth = j(d_pth, p, s)
            s_lst = sorted([int(prefix.split('.')[0]) for prefix in l(s_pth) ])

            for f in s_lst:
                f_pth = j(d_pth, p, s, str(f))
                scans.append(f_pth+".jpeg")
            img_paths.append(scans)
            
    return img_paths


def preprocess_jpeg(jpeg_path):
    
    img = cv2.imread(jpeg_path)
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
    
    return greyscale


def select_elements_with_spacing(input_list, divsion):
      
    spacing = len(input_list) // divsion
    if spacing == 0 :
        spacing = 1

    selected_indices = [spacing * i for i in range(0,divsion-1)]
    selected_indices.append(len(input_list)-1)

    selected_elements = [input_list[index] for index in selected_indices]
    
    return selected_elements