# VDL_Homework1
## Introduction
This is the HW1 in visual deep learning. In this task, we should classify the given image in one of the 100 class. However, the training set is a little bit unbalanced forming a big issus for image classification. I apply ResNet-50 with pretrained weight on `torchvision` which is called `IMAGENET1K_V2`. The following table shows the hyper-parameters for our training:
![image](https://github.com/user-attachments/assets/1df34fc4-9c00-4879-a042-0b5c32e710e1)


## Project structure
- `main.py` is the main function for training
- `model.py` is the file describe the resnet-50 and the Linear classifier
- `data.py` describes how to build up the pytorch dataset and the processing methods and augmentation methods I used
- `exploration.ipynb` is the notebook doing the data exploration and generate the testing result

## How to run the code
- install the dependencies and activate the environment
  ```
  conda env create --file=environment.yaml
  conda activate DL-Image
  ```
- check the model size
  ```
  python model.py
  ```
- train the model (if use default parameter, just run the following code)
  ```
  python main.py --log_name "Your Name"
  ```

## Performance
![image](https://github.com/user-attachments/assets/55e430e6-b412-4b5b-89fa-bffc5e61aa41)
