'''
using the open cv we display some single row and see the output in a image format and the dataset is
Mnist_test.csv dataset
'''

#import all the librarys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from Minist_train_explanation_using_CV import t_array

#take the minist-test dataset
df=pd.read_csv("mnist_test.csv")

#variable X in a independent columns and dependent columns in y

X=df.iloc[ : , 1: ]

y=df.iloc[ : , 0]

#take the first row

t=X.iloc[ 5 , : ]

#convert that into a array

t_array=t.to_numpy()

values=t_array.reshape(28,28)

plt.imshow(values,cmap="green")
plt.show()
