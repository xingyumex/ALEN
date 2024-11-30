# ALEN

## 1. Overview

This repository contains the source code and supplementary materials for the paper titled **ALEN: An Adaptive Dual-Approach for Enhancing Uniform and Non-Uniform Low-Light Images**. This research focuses on low-light images enhancement.

![ALEN_Architecture](ALEN_ARCH.png)

1. opencv-python == 4.9.0.80
2. scikit-image == 0.22.0
3. numpy == 1.24.3
4. torch == 2.3.0+cu118
5. Pillow == 10.2.0
6. tqdm ==  4.65.0
7. natsort == 8.4.0
8. torchvision == 0.18.0+cu118

## 2. Inference
To test the model, follow these steps:

1. Download the code and pretrained weights from this [link](https://drive.google.com/drive/folders/1Wuj5s1mtm5SJDLl80ISBRzhIwnRw4K1Q).

3. Place your images to be enhanced in the ./1_Input directory.

4. Run the code with the following command:

   ```bash
   python inference.py

5. The enhanced images will be saved in the ./2_Output directory.


## 3. Datasets  
This section describes the datasets used to train and evaluate the performance of **ALEN: Adaptive Light Enhancement Network** for low-light image enhancement.

### 3.1. Training Datasets  
The following public datasets were used to train the **ALEN** model. These datasets contain images with global and local illumination variations, necessary for effective classification and enhancement:

| **Dataset** | **Description**                                 | **Number of Images**         | **Type**              | **Resources** |
|-------------|:-----------------------------------------------:|:----------------------------:|:---------------------:|:-------------:|
| **GLI**     | Global-Local Illumination                       | 2,000                        | Paired Classification |Dataset        |
| **HDR+**    |                                                 | 22,472                       | Paired Enhancement    |Paper/Dataset  |
| **SLL**     | Synthetic Low-Light                             | 500                          | Paired Enhancement    |Paper/Dataset  |
| **MIT**     | MIT-Adobe FiveK                                 | 5,000                        | Paired Enhancement    |Paper/Dataset  |


### 3.2. Evaluation Datasets  
To evaluate the overall performance and generalization ability of **ALEN**, we used various datasets representing real-world scenarios:

| **Dataset**       | **Description**                                 | **Number of Images**     | **Type**                 | **Resources** |
|-------------------|:-----------------------------------------------:|:------------------------:|:------------------------:|:-------------:|
| **DIS**           | Diverse Illumination Scene                      | 10                       | Unpaired Enhancement     |Dataset        |
| **LSRW**          | Large-Scale Real-World                          | 735                      | Paired Enhancement       |Paper/Dataset  |
| **UHD-LOL4k**     | Synthetic Low-Light                             | 735                      | Paired Enhancement       |Paper/Dataseta |
| **DICM**          |                                                 | 69                       | Unpaired Enhancement     |Paper/Dataset  |
| **LIME**          |                                                 | 10                       | Unpaired Enhancement     |[Paper](https://ieeexplore.ieee.org/abstract/document/7782813)/Dataset  |
| **MEF**           |                                                 | 17                       | Unpaired Enhancement     |Paper/Dataset  |
| **NPE**           |                                                 | 8                        | Unpaired Enhancement     |Paper/Dataset  |
| **TM-DIED**       |                                                 | 222                      | Unpaired Enhancement     |[Dataset](https://sites.google.com/site/vonikakis/datasets/tm-died)|
