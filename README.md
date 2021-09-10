# Frequency-Scaling-in-Data-Augmentation-for-Image-Classification

![ours](https://user-images.githubusercontent.com/77310264/132606113-baf39b95-96f6-4445-a3ab-09958d2af41d.png)

# Abstract
Modern deep neural networks(DNNs) are used in artificial intelligence (AI) applications, including computer vision, speech recognition, and robotics. However, large-scale datasets must be collected and annotated to avoid overfitting. Data augmentation(DA) is an excellent alternative to labor-intensive and expensive real-world data collection in fields with limited access, such as the medical domain. In this study, we propose Frequency Scaling Data Augmentation(FSDA) to improve the image classification performance, noise robustness, and localizability of a classifier trained on a small-scale dataset. In our implementation, we designed two processes; a mask generation process(MGP) and a pattern scaling process(PSP). Each process clusters similar spectra in the frequency domain to obtain frequency patterns and scale patterns by learning weights from frequency patterns. Our method improves performace consistently with the CIFAR-10, Reduced CIFAR-10, and Chest X-ray datasets. For Reduced CIFAR-10, we achieved a Top-1 error rate of 32.88\%, which is 0.84\% lower than the previous method. We significantly improved performance for the Chest X-Rays dataset(Accuracy of 3.61\%, Precision of 6.73\%, F1-Score of 1.06\%, and AUC of 3.26\%), corruption robustness, and localizability. 

# Pseudo Code
## Mask Generation Process(MGP)
![Screenshot from 2021-09-09 10-21-58](https://user-images.githubusercontent.com/77310264/132606613-14c54225-0730-490d-b212-3547eade06f2.png)

## Pattern Scaling Process(PSP)
![Screenshot from 2021-09-09 10-22-07](https://user-images.githubusercontent.com/77310264/132606617-f821fcd8-5e4a-49c1-99a5-6f7e6e5f0c0d.png)

# Contents
This directory includes the inference phase in main.py and our model in models for CIFAR-10. We implement our method on Pytorch1.8, Python3, and Ubuntu 18.04. 

# Requirements
Install PyTorch and other required python libraries with:
```
pip install -r requirements.txt
```

# Usage
```
python main.py --augment [AUGMENT]
```
The pretrained model is available in 
