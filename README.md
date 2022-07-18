# Acne Classification with Deep Learning
All packages and dependencies are included in the [requirements.txt](https://github.com/yinchuangsum/acne_demo/blob/master/requirements.txt). 

## Python Version
- 3.7

## Introduction
Without a doubt, most of teenages face acne problem. However, there isn't a guideline on how serious the acne is and what proper steps should be taken in order to cure the acne and prevent scars. Hence, acne classification is developed using deep learning. It is carried out using Resnet-18 model in this project. It is able to **classify acne seriousness** into:

  1. Normal
  2. Level 0
  3. Level 1
  4. Level 2

as shown below:

<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/172897371-43b293af-8c4a-46aa-bcaa-6e801dbdbed2.png" width="500"> </br>

## Dataset
250 HD images were being hand-picked for each classes from various internet sources. 
<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/149659973-5242ca18-e52c-491f-aabc-f1773b39cb21.png" width="500"> </br>

## Annotation
Data annotation is being carried out by separating dataset into 4 classes.
<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/149660036-e69fb470-9e30-4249-9b83-6a678866c157.png" width="500"> </br>

## Data Preprocessing
To increase the size of the dataset for training, data preprocessing is being carried out which includes:
- Flip:      		Horizontal, Vertical
- 90° Rotate: 	Clockwise, Counter-Clockwise, Upside Down
- Crop: 		    0% Minimum Zoom, 50% Maximum Zoom
- Rotation: 	  Between -15° and +15°
- Blur: 		    Up to 10px
- Rotate:		    30 degree

## Installling Requirements
```
pip install -r requirements.txt
```

## Run the code
```
streamlit run main.py
```

## Results of AI Model
The AI Model is able to achieve up to 90% accuracy by training with only 250 HD images from each classes. 

Accuracy             |  Loss Function |  Confusion Matrix
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/22144223/172899927-31fe9853-21f6-4562-bb02-7874ee5ead0f.png" width="400"> | <img src="https://user-images.githubusercontent.com/22144223/172898955-6d6467ab-6d6c-4d9e-a142-7be887d74401.png" width="400"> | <img src="https://user-images.githubusercontent.com/22144223/179527348-045a6f02-a2ca-44df-ac2d-cb550944d87b.png" width="400">



## Future Improvements
1. Platform to discuss skin care products
2. Cross geographical skin samples
3. Develop smartphone app
4. More detailed classifier
5. Higher Accuracy

## Additional Information
Additional information about this project can be read [here](https://docs.google.com/presentation/d/1f4I75eh2MxlMGislCCR2iAQR7PfxG1FbuIr79Q8eg30/edit?usp=sharing).
