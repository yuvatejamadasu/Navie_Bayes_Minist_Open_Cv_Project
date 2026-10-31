'''
take the minist dataset and see the image for each of them using the open computer vision
'''

#import all the librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sklearn

#take the minist_train dataset

df=pd.read_csv("mnist_train.csv")


#X in a independent column and y in dependent column

X=df.iloc[ : , 1: ]

y=df.iloc[ : , 0]


#take only the first row and see the output as a image using the open cv
#take only the second row and see the output as a image using the opoen cv

t=X.iloc[ 5 , : ]


#give this first row to the opencv this shows a image in the output

#convert first into a numpy

t_array=t.to_numpy()


#when we give our data in a opencv it must me a 2D so convert the t_array in 2D

#in our data total indpendent columns is 784 so connvert 28 x 28

values=t_array.reshape(28,28)
print(values)
#now the data is 2D give this data to openCV

plt.imshow(values,cmap="gray")

plt.show()



