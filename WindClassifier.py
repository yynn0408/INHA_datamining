
# # 와인 감별사 : 와인의 Quality를 분류하는 Classifier 만들기


import keras

keras.__version__

from keras import Sequential
from keras.layers import Dense, Activation

import pandas as pd
pd.__version__
pd.options.display.max_rows=15


# In[2]:


import numpy as np
np.__version__
white_wine=pd.read_csv('C:\\Users\\82109\\Desktop\\PPT\\wine data\\winequality-white.csv',sep=',',dtype='unicode')
red_wine=pd.read_csv('C:\\Users\\82109\\Desktop\\PPT\\wine data\\winequality-red.csv',sep=',',dtype='unicode')

white_wine.dtypes
display(white_wine)



from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
def preprocess(df):
    Scaler=StandardScaler()
    data=df.iloc[:,:-1]
    qual=df.iloc[:,-1]
    Scaler.fit(data)
    df=Scaler.transform(data)
    df=pd.DataFrame(df)
    qual=pd.DataFrame(qual)
    df=pd.concat([df,qual],axis=1)
    return df
    
display(preprocess(white_wine))
#display(white_wine)
preprocess(white_wine)


# In[7]:


display(preprocess(red_wine))
preprocess(red_wine)
def generate_data(df, t_r):
    x_data=df.iloc[:,:-1]
    y_data=df.iloc[:,-1]
    X_train=x_data.sample(frac=t_r,random_state=0)#random_state=seed
    Y_train=y_data.sample(frac=t_r,random_state=0)#random_state=seed
    X_test=x_data.drop(X_train.index)
    Y_test=y_data.drop(Y_train.index)
    Y_train=keras.utils.to_categorical(Y_train,num_classes=None)
    Y_test=keras.utils.to_categorical(Y_test,num_classes=None)
    return X_train.values, Y_train, X_test.values, Y_test

#####################################################



x_train, y_train, x_test, y_test = generate_data(white_wine, 0.7)


###########################################################
import matplotlib.pyplot as plt
##rmsprop<-adam 성능 증가#lr=0.001로 하니 너무 느리게 수렴->0.01로 하면 너무 ㅐ
class whwinemodel(object):#,node,opt,loss,learn,act
    def __init__(self):
        self.model=Sequential()
    def construct(self):
        self.model.add(Dense(32,activation='relu',input_dim=11)) 
        self.model.add(Dense(25,activation='relu'))
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dense(10,activation='softmax'))
        sgd=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)  
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])#adam,rmsprop각각
    def fit(self):
        self.history=self.model.fit(x_train,y_train,epochs=300,batch_size=64,verbose=1)#validationset만드든것 성능 비교
        
    def printkey(self):
        print(self.history.history.keys())
        
    def figure(self):
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.title('test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax2=fig.add_subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(x_train, y_train, verbose=0)
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
    def test(self):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("test")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
firstmodel=whwinemodel()
firstmodel.construct()
firstmodel.fit()
firstmodel.figure()
firstmodel.test()



class redwinemodel(object):#,node,opt,loss,learn,act
    def __init__(self):
        self.model=Sequential()
    def construct(self):
        self.model.add(Dense(32,activation='relu',input_dim=11)) 
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dense(9,activation='softmax'))
        sgd=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)  
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])#adam,rmsprop각각
    def fit(self):
        print(ry_train.shape)
        self.history=self.model.fit(rx_train,ry_train,epochs=250,batch_size=64,verbose=1)#validationset만드든것 성능 비교
        #128로 했을 때 성능 더 굳 256보다
       # self.history=self.model.train_on_batch(x_train,y_train,sample_weight=128,)
    def printkey(self):
        print(self.history.history.keys())
        
    def figure(self):
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.title('test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax2=fig.add_subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(rx_train, ry_train, verbose=0)
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
    def test(self):
        score = self.model.evaluate(rx_test, ry_test, verbose=0)
        print("test")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))

firstmodel=whwinemodel()
firstmodel.construct()
firstmodel.fit()
firstmodel.figure()
firstmodel.test()
                
rx_train, ry_train, rx_test, ry_test = generate_data(red_wine, 0.7)        
redfirstmodel=redwinemodel()
redfirstmodel.construct()
redfirstmodel.fit()
redfirstmodel.figure()
redfirstmodel.test()

###########################################################


# ### 2. 각 모델의 성능을 향상시킬 수 있는 방법 적용


###########################################################minmaxscaler성능 뱃
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization,regularizers

class whwinemodel(object):#,node,opt,loss,learn,act
    def __init__(self):
        self.model=Sequential()
    def construct(self):
        self.model.add(BatchNormalization())
        self.model.add(Dense(512,input_dim=11))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(keras.layers.core.Dropout(0.2))
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(128))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())#지워보기
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))#64개인거로 했을떄 성능은 비슷하거나 더 안좋았음 
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(10,activation='softmax'))#rms,adagrad,adadelta
        sgd=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])#adam,rmsprop각각
    def fit(self):
        self.history=self.model.fit(x_train,y_train,epochs=200,batch_size=256,verbose=1,validation_data=(x_test,y_test))#validationset만드든것 성능 비교
        #128로 했을 때 성능 더 굳 256보다
       # self.history=self.model.train_on_batch(x_train,y_train,sample_weight=128,)
    def printkey(self):
        print(self.history.history.keys())
        
    def figure(self):
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.title('test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax2=fig.add_subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(x_train, y_train, verbose=0)
        print("training")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
    def test(self):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("test")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        



class redwinemodel(object):#,node,opt,loss,learn,act
    def __init__(self):
        self.model=Sequential()
    def construct(self):
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(keras.layers.core.Dropout(0.2))#이거만 냅두기
        self.model.add(Dense(128))#64개인거로 했을떄 성능은 비슷하거나 더 안좋았음 
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))#64개인거로 했을떄 성능은 비슷하거나 더 안좋았음 
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))#64개인거로 했을떄 성능은 비슷하거나 더 안좋았음 
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
      #  self.model.add(Dense(20))
      #  self.model.add(BatchNormalization())
     #   self.model.add(Activation('relu'))
        self.model.add(Dense(10,activation='softmax'))#rms,adagrad,adadelta
        sgd=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#성능 굳굳...ㄸ
    #    sgd=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)  
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])#adam,rmsprop각각
    def fit(self):
        self.history=self.model.fit(x_train,y_train,epochs=300,batch_size=256,verbose=1,validation_data=(x_test,y_test))#validationset만드든것 성능 비교
        #128로 했을 때 성능 더 굳 256보다
       # self.history=self.model.train_on_batch(x_train,y_train,sample_weight=128,)
    def printkey(self):
        print(self.history.history.keys())
        
    def figure(self):
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.title('test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax2=fig.add_subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(x_train, y_train, verbose=0)
        print("training")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
    def test(self):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print("red test")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
redmodel=redwinemodel()
redmodel.construct()
redmodel.fit()
redmodel.figure()
redmodel.test()



###########################################################




##########################################################
data=white_wine.append(red_wine)
xt_train, yt_train, xt_test, yt_test = generate_data(data, 0.7)
class wrwinemodel(object):#,node,opt,loss,learn,act
    def __init__(self):
        self.model=Sequential()
    def construct(self):
        self.model.add(BatchNormalization())
        self.model.add(Dense(256,activation='relu',input_dim=11))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(128))
        self.model.add(keras.layers.core.Dropout(0.2))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(10,activation='softmax'))#rms,adagrad,adadelta
        sgd=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)#성능 굳굳...ㄸ
        self.model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])#adam,rmsprop각각
    def fit(self):
        self.history=self.model.fit(xt_train,yt_train,epochs=300,batch_size=128,verbose=1,validation_data=(xt_test,yt_test))#validationset만드든것 성능 비교
        #128로 했을 때 성능 더 굳 256보다
       # self.history=self.model.train_on_batch(x_train,y_train,sample_weight=128,)
    def printkey(self):
        print(self.history.history.keys())
        
    def figure(self):
        fig=plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.plot(self.history.history['acc'])
        plt.title('test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax2=fig.add_subplot(1,2,2)
        plt.plot(self.history.history['loss'])
        #plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()
        
    def evaluate(self):
        score = self.model.evaluate(xt_train, yt_train, verbose=0)
        print("training")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
    def test(self):
        score = self.model.evaluate(xt_test, yt_test, verbose=0)
        print("test")
        print("eveluate loss:{}, evaluate acc:{}".format(score[0],score[1]))
        
wrmodel=wrwinemodel()
wrmodel.construct()
wrmodel.fit()
wrmodel.figure()
wrmodel.test()


