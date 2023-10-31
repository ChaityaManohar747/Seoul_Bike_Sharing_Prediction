import pandas as pd
from sklearn.model_selection import train_test_split

class Dataloader(): 

    def __init__(self,data,features,target):
       
        self.data = data
        self.input_data = self.data[features]
        self.target_data = self.data[target]


    def getHeader(self):
        return list(self.data.columns.values)

    def getData(self):
        X_train,X,Y_train,Y = train_test_split(self.input_data,self.target_data,test_size=0.6)
        X_val,X_test,Y_val,Y_test = train_test_split(X,Y,test_size=0.5)

        return X_train,Y_train,X_val,Y_val,X_test,Y_test

    def getFullData(self):
        return self.data