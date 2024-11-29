# ALEN

## Overview

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

## Inference
To test the model, follow these steps:

1. Download the code and pretrained weights from this [link](https://drive.google.com/drive/folders/1Wuj5s1mtm5SJDLl80ISBRzhIwnRw4K1Q).

3. Place your images to be enhanced in the ./1_Input directory.

4. Run the code with the following command:

   ```bash
   python inference.py

5. The enhanced images will be saved in the ./2_Output directory.
