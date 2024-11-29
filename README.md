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

| **Dataset** | **Description**                              | **Number of Images** |
|-------------|:----------------------------------------------:|:----------------------:|
| **DICM**    | Dataset for Image Contrast Enhancement       | 300                  |
| **LIME**    | Low-light Image Enhancement Dataset          | 200                  |
| **DIS**     | Diverse Illumination Scenes                  | 500                  |
| **MEF**     | Multi-Exposure Fusion Dataset                | 350                  |
| **NPE**     | Naturalness-Preserved Enhancement Dataset    | 400                  |


### 3.2. Evaluation Datasets  
To evaluate the overall performance and generalization ability of **ALEN**, we used various datasets representing real-world scenarios:

- **EUVP**: Underwater Image Dataset (500 underwater images)  
- **LOL**: Low-Light Image Dataset (485 images captured in low-light conditions)  
- **RUIE**: Real-World Underwater Image Enhancement Dataset (600 real underwater images)
