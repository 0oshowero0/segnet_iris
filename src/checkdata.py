# _*_ coding: utf-8 _*_


import os.path
from PIL import Image
import numpy as np
def main():
    origin_data_root="../data/iris_segmentation/images/"
    origin_label_root="../data/iris_segmentation/labels/"

    for root, dirs, files in os.walk(origin_data_root):
        for filename in files:
                img = Image.open(origin_data_root+filename)
                label = Image.open(origin_label_root+filename)
                if img.mode != 'L' or img.size != (1024,1024) or label.mode != 'L' or label.size != (1024,1024):
                    print('./'+filename)
"""                label_data = np.array(label.getdata(),dtype=int)
                 for pix in label_data:
                    if pix !=0 and pix != 255:
                        print(filename)
                        break """
                

   

if __name__ == "__main__" :
    main()