import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skm 
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder


def expand_feature(data,col,feature):
    data[col] = feature[:,0]
    col_ind = data.columns.get_loc(col) + 1
    for f in range(1,feature.shape[1]):
        data.insert(col_ind,col+'_'+str(f),feature[:,f])
        col_ind+=1

def make_categorical_onehot(data,test_data):
    categorical_cols=[c for c in data.columns if (1<data[c].nunique()) & (data[c].dtype != np.number)& (data[c].dtype != int)]
    label_encoder = LabelEncoder()
    for col in categorical_cols:

        label_encoder = label_encoder.fit(data[col])
        feature = np.expand_dims(label_encoder.transform(data[col]),1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)

        test_feature = np.expand_dims(label_encoder.transform(test_data[col]),1)
        test_feature = onehot_encoder.transform(test_feature)

        expand_feature(data,col,feature)
        expand_feature(test_data,col,test_feature)

    return data, test_data

def remove_majority_missing(data,test_data):
    maj_missing_cols = [c for c in data.columns if data[c].isna().sum()/len(data) > .5]
    return data.drop(maj_missing_cols,axis=1), test_data.drop(maj_missing_cols,axis=1)

def fill_in_missing(data): #not ideally done
    cols_with_missing = [col for col in data.columns 
                                 if data[col].isnull().any()]
    for col in cols_with_missing:
        if data[col].dtype == np.int64:
            data[col]=data[col].replace(np.nan,data[col].mode())
        elif data[col].dtype == np.float64:
            data[col]=data[col].replace(np.nan,data[col].mean())
    return data

def hand_select_columns(data, test=False):
    hand_selected = ['facility_type', 'floor_area', 'year_built', 'energy_star_rating', 'cooling_degree_days', 'heating_degree_days', 'max_temp', 'min_temp', 'precipitation_inches','site_eui','id']

    #add max of max temp and min of min temp.
    max_cols = [col for col in data.columns if 'max_temp' in col]
    data['max_temp'] = data[max_cols].max(axis=1)

    min_cols = [col for col in data.columns if 'min_temp' in col]
    data['min_temp'] = data[min_cols].min(axis=1)

    if test:
        hand_selected.remove('site_eui')
    return data[hand_selected]

def clean_data(data,test_data):
    data = hand_select_columns(data)
    test_data = hand_select_columns(test_data, test=True)

    data, test_data = remove_majority_missing(data,test_data)

    data, test_data = make_categorical_onehot(data,test_data)

    data = fill_in_missing(data)
    test_data = fill_in_missing(test_data)

    return data.iloc[:,:-1], test_data #removing id column from train

def df_to_XY_tuple(data, inds, test=False): #only for train data
    return (np.array(data.iloc[inds,0:-1]), np.array(data.iloc[inds,-1]))

def make_train_val(data, val=.2, seed=444):
    np.random.seed(seed)
    tot_num = len(data)
    val_inds = np.random.choice(tot_num,int(tot_num*val),replace=False)
    train_inds = np.setdiff1d(np.arange(tot_num),val_inds)

    return df_to_XY_tuple(data,train_inds), df_to_XY_tuple(data,val_inds)

def run_xgb_opt(X_train,Y_train):
    '''performs xgb model grid search and returns best score and parameters'''
    xgb_model = xgb.XGBRegressor(eval_metric='rmse')
    clf = GridSearchCV(xgb_model,
                   {'max_depth': [5,6,7],
                    'n_estimators': [600,800, 900]}, verbose=2, n_jobs=1, cv=2, 
                    scoring = skm.make_scorer(skm.mean_squared_error)) 
    clf.fit(X_train,Y_train)
    return clf.best_score_, clf.best_params_

def train_early_stop(model,train,val):
    '''trains model using a validation set for early stopping'''
    print('Using early stopping:')
    
    eval_set = [val]
    model.fit(train[0],train[1],verbose=1,eval_set=eval_set,early_stopping_rounds=50);
    return model


def get_column(df,inds,col):
    '''returns numpy array of values from given named column and row indices'''
    col_data = df[col]
    return np.array(col_data.iloc[inds])

    




