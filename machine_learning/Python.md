##load data

pd.DataFrame(data , columns)

## train 
standardScaler = StandardScalar()
pricescaler =standardScaler.fit(price1)
scaled_price =pricescalar.transform( price1)
extreme_grad_boost_model = xgboost.XGBRegressor ()
extreme_grad_boost_model.fit( scaled_price, price1)


## generate model

## save model

pickle.dump( extreme_grad_boost_model, "extreme_grad_boost_model.sav")
joblib.dump( pricescaler, "pricescaler.sav")

## remove outliers
for col in cols :
    updated = if Val > 70000 , Val = 70000

## prediction 

price_model = LinearRegression.fit( scaled_price )
price_model.predict ( scaled_Val )
