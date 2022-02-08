# LG-AI-Challenge-for-Plant-Classification
Dacon에서 진행된 [농업 환경 변화에 따른 작물 병해 진단 AI 경진대회](https://dacon.io/competitions/official/235870/overview/description)
에 대한 코드입니다.


## Requirements
* python==3.7.12
* albumentations==1.1.0
* numpy==1.19.5
* pandas==1.3.5
* cv2==4.1.2
* sklearn==1.0.2
* json==2.0.9
* torch==1.10.0+cu111
* timm==0.5.4
* transformers==4.16.2


## Preprocessing(image)
* Augmentations: Transpose, Flip, Rotate, RandomBrightnessContrast, Cutmix ...

## Preprocessing(sequence)
* Augmentations : 만약 길이가 500이상이라면 random으로 sampling(2씩)
* MinMax Scaling

## Model
* resnext50_32x4d 

## Training
* K-fold Cross Validation(k=5)
* Use Cutmix augmentation until epoch 15
* Save model's weight when score is highest

## inference 
* K-fold Model Ensemble(Soft Voting)
