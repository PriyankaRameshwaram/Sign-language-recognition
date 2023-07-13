#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Import our libraries

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


#Get our training and test data

train = pd.read_csv(r"C:\Users\rohit\Downloads\dataset_mnist\sign_mnist_train.csv")
test = pd.read_csv(r"C:\Users\rohit\Downloads\dataset_mnist\sign_mnist_test.csv")


# In[31]:


#inspecting train data

train.head()


# In[32]:


train.shape


# In[33]:


test.shape


# In[34]:


#Get our training labels

labels = train['label'].values
labels


# In[35]:


#View the unique label 24 in total

unique_val = np.array(labels)
print(unique_val)
np.unique(unique_val)


# In[36]:


#Plot the quanities in each graph

plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# In[37]:


#Drop training labels from our training data so we can separate it

train.drop('label',axis=1,inplace = True)


# In[38]:


train.head()


# In[39]:


train.values.shape


# In[40]:


#Extract the image from data for each row in our csv,it's in rows of 784 coloumns

images = train.values
images = np.array([np.reshape(i,(28,28)) for i in images])
images = np.array([i.flatten() for i in images])


# In[41]:


#One hot encoding 

from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels


# In[42]:


#Inspect an image

index = 5
print(labels[index])
plt.imshow(images[index].reshape(28,28))


# In[43]:


# Use openCV to view 10 random images from our training data

import cv2
import numpy as np
for i in range(0,10):
    rand = np.random.randint(0,len(images))
    input_im = images[rand]
    sample = input_im.reshape(28,28).astype(np.uint8)
    sample = cv2.resize(sample,None,fx=10,fy=10,interpolation = cv2.INTER_CUBIC)
    cv2.imshow("sample image",sample)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# In[44]:


#Split our data into x_train,x_test,y_traion,y_test

from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(images , labels , test_size = 0.3 , random_state = 101)


# In[45]:


#Start loading our tensorflow module and define batch size

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
batch_size = 128
num_classes = 24
epochs = 10


# In[46]:


#Scale our images

x_train = x_train/255
x_test = x_test/255


# In[47]:


#Reshape them into size required by tf,keras

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

plt.imshow(x_train[0].reshape(28,28))


# In[48]:


#CNN Model

from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(64, kernel_size = (3,3),activation ='relu' , input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(num_classes, activation='softmax'))



# In[49]:


model.compile(loss = 'categorical_crossentropy',optimizer = Adam(), metrics = ['accuracy'])


# In[50]:


print(model.summary())


# In[ ]:


#Train our model

history = model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs=epochs,batch_size=batch_size)


# In[ ]:


#Save our model

model.save("sign_mist_cnn_50_Epochs.h5")
print("Model Saved")


# In[26]:


import warnings
warnings.filterwarnings("ignore")


# In[27]:


#View our training history graphically
#ACCURACY

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Evaluation")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend('train','test')

plt.show()


# In[28]:


#LOSS

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("loss Evaluation")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend('train','test')

plt.show()


# In[29]:


test = pd.read_csv(r"C:\Users\rohit\Downloads\dataset_mnist\sign_mnist_test.csv")

test.head()


# In[30]:


#Reshape our test data so that we evaluate it's performance on an unseen data

test_labels = test['label']
test.drop('label',axis=1,inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i,(28,28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0],28,28,1)
test_images.shape
y_pred = model.predict(test_images)


# In[31]:


y_pred[:1]


# In[32]:


len(y_pred)


# In[33]:


len(test_labels)


# In[34]:


# label.shape
test_labels.shape


# In[35]:


test.shape


# In[36]:


test_labels= test_labels[:len(y_pred)]


# In[37]:


# Get our accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(test_labels , y_pred.round())


# In[38]:


#Create function to match label to letter

def getLetter(result):
    classLabels = {0: 'A',
                  1:'B',
                  2:'C',
                  3:'D',
                  4:'E',
                  5:'F',
                  6:'G',
                  7:'H',
                  8:'I',
                  9:'K',
                  10:'L',
                  11:'M',
                  12:'N',
                  13:'O',
                  14:'P',
                  15:'Q',
                  16:'R',
                  17:'S',
                  18:'T',
                  19:'U',
                  20:'V',
                  21:'W',
                  22:'X',
                  23:'Y'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"


# In[39]:



plt.imshow(x_test[0].reshape(28,28))


# In[40]:


x = x_test[0].reshape(1,28,28,1)


# In[43]:


result = str(model.predict_classes(x))
print(f'The Predicted class for the random image is:{getLetter(result[1:2])}')


# # TEST OUR MODEL WITH WEBCAM

# In[ ]:



cap = cv2.VideoCapture(0)
while True:
    
    ret, frame = cap.read()
    roi = frame[100:400, 320:620]
    cv2.imshow('roi',roi)
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)
    
    cv2.imshow('roi sacled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy,(320,100), (620,400),(250,0,0), 5)
    
    roi = roi.reshape(1,28,28,1)
    
    result = str(model.predict_classes(roi, 1, verbose = 0)[0])
    cv2.putText(copy,getLetter(result),(300,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow('frame', copy)
    
    if cv2.waitKey(1) == 13:
        break
        
cap.release()
cv2.destroyAllWindows()
    
    

