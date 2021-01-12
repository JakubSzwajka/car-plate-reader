# License Plate recognition

## Instalation 
1. install dependencies from requirements.txt
2. use: ```pip install -r requirements.txt``` 
2. specify paths in settings.py // model, photos etc. 

## Run 
* check path TEST_IMAGE_PATH in settings.py. The default image should be: 
## ![Test car](/photos/car3.jpg)
* run script main.py 
* check your console. You should get something like 
## ![console terminal](/screenshots/screen.png)

## Known Problems
* When program cant find letters try to change ratio values. 
Plotter.py lines 60 - 61

## Info 
* Based on this [article](https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922) . 