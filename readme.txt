*****************************************************************************************************************************************************
							GUIDE TO RUN THE PROJECT
*****************************************************************************************************************************************************

1. Open the "code" folder and move the "vgg16_generalized.ipynb" to your Jupyter notebook. Place this jupyter source file under a new folder. (You can keep any name to this newly created folder)

2. Open the "dataset" folder and place all the images (icons + test images) in a new folder named "data" directly under C drive in local machine. (C:\data)

3. Run the "vgg16_generalized.ipynb" file in Jupyter notebook. The libraries need to be installed are: Tensorflow, Keras, NumPy and Matplolib. Following are the commands:

		from tensorflow.keras.applications import VGG16
		from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
		from tensorflow.keras.models import Model
		from tensorflow.keras.optimizers import Adam
		from tensorflow.keras.preprocessing.image import ImageDataGenerator
		from tensorflow.keras.callbacks import EarlyStopping
		import numpy as np
		import matplotlib.pyplot as plt

4. After successful running of the "vgg16_generalized.ipynb".

5. The model will be saved as "vgg16.h5" file.

6. Now give the path of "vgg16.h5" file in "app.ipynb" and run it. Import the necessary libraries before running using following commands:
		import streamlit as st
		from PIL import Image
		import numpy as np
		import tensorflow as tf
		import matplotlib.pyplot as plt
		import cv2
		from mpl_toolkits.mplot3d import Axes3D  

7. Now run the "app.py" file through the command line prompt.To run the file install streamlit and googleprob.
		
command-> streamlit run app.py

8. Local host interface will be opened -> upload or capture the image and press the predict button. 

9. The image will be predicted whether given image is "Hand-Made" or "Machine-Made" with its accuracy.
   It will also display the pie chart of prediction.

NOTE: Change the paths of files and folders in the code according to the files saved in your system.And install all the libraries mentioned.