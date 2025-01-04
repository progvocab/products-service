## dependency libraries 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler,MinMaxScalar, MaxAbsScalar 

import xgboost 

import pickle 

## load data

df = pd.DataFrame(data , columns)

## train 

X = df["price"]
y = df["rating"]

X_train , X_test , y_train , y_test = train_test_split ( X,y, random_state =42, test_size = 0.25 , shuffle=True )

- train set : will be utilized to fit the model , training data (  learning data ) for model 
- test set : subset of training data for accurate evaluation of model fit
- validation set : data from training data for evaluating model performance 


standardScaler = StandardScalar()
pricescaler =standardScaler.fit(price1)
scaled_price =pricescalar.transform( price1)
extreme_grad_boost_model = xgboost.XGBRegressor ()
extreme_grad_boost_model.fit( scaled_price, price1)




## save model

pickle.dump( extreme_grad_boost_model, "extreme_grad_boost_model.sav")
joblib.dump( pricescaler, "pricescaler.sav")

## remove outliers
for col in cols :
    updated = if Val > 70000 , Val = 70000

## prediction 

price_model = LinearRegression()
price_model.fit( train_X , train_y  )
price_prediction = price_model.predict ( scaled_Val )
