import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

df=pd.read_csv("train.csv")
images=os.listdir("train_images")
img_list=[]
img_list1=[]
for i in images:
    image=cv2.imread("train_images\\"+i)
    image=cv2.resize(image,(400,400))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_list1.append(image)
    kopya=image.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)
    Gaussian_blur=cv2.GaussianBlur(kopya,(5,5),0)
    Thershold=cv2.threshold(Gaussian_blur,10,255,cv2.THRESH_BINARY)[1]
    kontur=cv2.findContours(Thershold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    kontur=kontur[0][0]
    kontur=kontur[:,0,:]
    x1=tuple(kontur[kontur[:,0].argmin()])[0]
    y1=tuple(kontur[kontur[:,1].argmin()])[1]
    x2=tuple(kontur[kontur[:,0].argmax()])[0]
    y2=tuple(kontur[kontur[:,1].argmax()])[1]
    x=int(x2-x1)*4//50
    y=int(y2-y1)*5//50
    kopya2=image.copy()
    if x2-x1>100 and y2-y1>100:
        kopya2=kopya2[y1+y:y2-y,x1+x:x2-x]
        kopya2=cv2.resize(kopya2,(400,400))
    lab=cv2.cvtColor(kopya2,cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)
    clahe=cv2.createCLAHE(clipLimit=5.0,tileGridSize=((8,8)))
    cl=clahe.apply(l)
    limg=cv2.merge((cl,a,b))
    son=cv2.cvtColor(limg,cv2.COLOR_LAB2RGB)
    med_son=cv2.medianBlur(son,3)
    arka_plan=cv2.medianBlur(son,39)
    maske=cv2.addWeighted(med_son,1,arka_plan,-1,255)
    son_img=cv2.bitwise_and(maske,med_son)
    img_list.append(son_img)


y_train=pd.get_dummies(df["diagnosis"]).values
y_train_son=np.ones(y_train.shape,dtype="uint8")
y_train_son[:,4]=y_train[:,4]

for i in range(3,-1,-1):
    y_train_son[:, i] = np.logical_or(y_train[:, i], y_train_son[:, i + 1])

x_train=np.array(img_list)


from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train_son,test_size=0.15,random_state=2019,shuffle=True)
print("x_train_boyut :",x_train.shape)
print("x_val_boyut :",x_val.shape)
print("y_train_boyut :",y_train.shape)
print("y_val_boyut :",y_val.shape)

from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
data_generator=datagen.flow(x_train,y_train,batch_size=2,seed=2020)

from efficientnet.keras import EfficientNetB7
#örnek_model=EfficientNetB5()
#örnek_model2=EfficientNetB5(include_top=False)
from keras.models import Sequential
from keras import layers
model=Sequential()
#model.add(EfficientNetB7(include_top=False))
model.add(EfficientNetB7(weights="imagenet",include_top=False, input_shape=(400,400,3)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5,activation="sigmoid"))

from keras.optimizers import Adam
model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.00005),metrics=["accuracy"])

from keras.callbacks import ReduceLROnPlateau
lr=ReduceLROnPlateau(monitor="val_loss",patience=3,verbose=1,mode="auto",factor=0.25,min_lr=0.000001)
history=model.fit_generator(data_generator,steps_per_epoch=10,epochs=5,validation_data=(x_val,y_val), callbacks=[lr])






























