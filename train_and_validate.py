import pandas as pd
import os
import xgboost as xgb
import sklearn.metrics as skm 
from model_funcs import *
from datetime import datetime

#Prepping data
base_path = '/home/grace/cluster_code/climate/WiDS_datathon'
data = pd.read_csv(base_path+'/train.csv')
test_data = pd.read_csv(base_path+'/test.csv')

data, test_data = clean_data(data, test_data)
print(data);

train, val = make_train_val(data)

#Optimize hyper-parameters
best_score, best_params = run_xgb_opt(train[0],train[1])
print(best_params)

#Train
best_params['eval_metric'] = 'rmse'
xgb_model = xgb.XGBRegressor(**best_params)
xgb_model = train_early_stop(xgb_model,train,val)

train_pred = xgb_model.predict(train[0])
print('\nPerformance on training data') 
print('RMSE: %.2f' % np.sqrt(skm.mean_squared_error(train[1],train_pred))) 

val_pred = xgb_model.predict(val[0])
print('\nPerformance on validation data') 
print('RMSE: %.2f' % np.sqrt(skm.mean_squared_error(val[1],val_pred))) 

#actually run on test data
test_pred = xgb_model.predict(np.array(test_data.iloc[:,:-1]))
#create df and write to csv
test_results = test_data.loc[:,['id']]
test_results.insert(1,'site_eui',test_pred)

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
test_results.to_csv(base_path+'/results/results'+date_time+'.csv',index=False)



