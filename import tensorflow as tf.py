import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("Housing Price data set.csv")
print(data.shape)
print(data.info())
print(data)
for i in data:
    if(str(data[i].dtype)=="object"):
        data[i]=LabelEncoder().fit_transform(data[i])
x=data.iloc[:,1:]
y=data.iloc[:,0]
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.3,random_state=42)
linearmodel=LinearRegression().fit(trainx,trainy)
ypredict=linearmodel.predict(testx)
print('root mean square error is ',np.sqrt(mean_squared_error(testy, ypredict)))
print('r2 score',r2_score(testy,ypredict))