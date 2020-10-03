# Machine Learning for Pneumonia chest X-ray
this repository applies is machine learning application to detect pneumonia in chest X-ray images. it is implemented using the Keras library (imported for the Tensorflow library).\
\
The images used for training and testing were taken form: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
\
\
to use the application download the images form the link above and extract the chest_xray directory into a directory named "images", your directory should look as so:\
```
images
    |__chest_xray
        |__test
        |__train 
```
after doing so, install the requirments by running the following command:
```commandline
pip install -r requirements.txt
``` 
once the installation is complete, start by running ```rescale_load.py``` to rescale the images to the proper size and load them into ```.pickle``` files.

afterwords you can run  ```model_train.py``` to train the model followed by ```test_accuracy.py``` to test the trained model (note ```model_train.py``` will also create tensorboard log files).\
\
\
Future additions:
- replace hardcoded variables for commandline arguments.