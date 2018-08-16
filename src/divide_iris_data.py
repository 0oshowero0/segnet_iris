# _*_ coding: utf-8 _*_


import os.path
import pandas as pd
import numpy as np
def main():
    origin_data_root="../data/iris_segmentation/images"
    output_fname_t="../data/iris_segmentation/lists/train.csv"
    output_fname_e="../data/iris_segmentation/lists/eval.csv"

    train_img = []
    train_label = []
    eval_img = []
    eval_label = []

    for root, dirs, files in os.walk(origin_data_root):
        for filename in files:
            if (np.random.random()>0.3):
                train_img.append("../data/iris_segmentation/images/"+filename)
                train_label.append("../data/iris_segmentation/labels/"+filename)
            else:
                eval_img.append("../data/iris_segmentation/images/"+filename)
                eval_label.append("../data/iris_segmentation/labels/"+filename)

    train_csv = pd.DataFrame({'images':train_img,'labels':train_label})

    train_csv.to_csv(output_fname_t,header=None,index=None)

    eval_csv = pd.DataFrame({'images':eval_img,'labels':eval_label})

    eval_csv.to_csv(output_fname_e,header=None,index=None)


                    





   

if __name__ == "__main__" :
    main()