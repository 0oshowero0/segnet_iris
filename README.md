# Iris Segmentation Using Segnet

This repo borrows codes from
https://github.com/meetshah1995/pytorch-semseg
and https://github.com/utkuozbulak/pytorch-custom-dataset-examples.

Given that Iris dataset has been downloaded as "Iris_Segmentation_Dataset.tar.gz".

1. Make a new dir named _"data"_

    ```bash
    # From segnet_iris/
    mkdir data
    ```

2. Extract the dataset into _data_ folder

    ```bash
    # From segnet_iris/data/
    tar -zvxf Iris_Segmentation_Dataset.tar.gz
    ```

3. Copy _eval.csv_ and _train.csv_ from _segnet_iris_ to _segnet_iris/data/iris_segmentation/lists/_

   (you may also run _divide_iris_data.py_ from _src_ to generate new lists)

4. For training, run _train.py_
    ```bash
    # From segnet_iris/src/
    python train.py
    ```

5. For evaluation, modify _eval.py_ according to the best model you have got. Then make a dir named _output_ in _segnet_iris_. Run _eval.py_.

    ```bash
    # From segnet_iris/src/
    python eval.py
    ```