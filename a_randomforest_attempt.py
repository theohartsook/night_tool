from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.utils import resample
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def prepRFData(input_df, drop_cols=['2_uid', '3_uid', '4_uid', '5_uid', 'uid'], max_xy_dist=2, max_r_diff=1., xy_dist='gt_xy_dist', r_diff='gt_r_diff', det_class='gt_uid'):
    """ This is a convenience function to clean up my predictions for Random
        Forest. """    
    print('prep')    
    input_df.drop(columns=drop_cols, inplace=True)
    input_df.fillna(1000, inplace=True)

    
    print(input_df.gt_uid.value_counts())
    #input_df.loc[input_df[xy_dist] > max_xy_dist*input_df['r'], det_class] = 0
    #input_df.loc[input_df[r_diff] > max_r_diff*input_df['r'], det_class] = 0
    #input_df.loc[input_df[xy_dist] > (input_df['r'] + 2*np.sqrt(input_df['r'])), det_class] = 0
    #input_df.loc[input_df[r_diff] > (input_df['r'] + 2*np.sqrt(input_df['r'])), det_class] = 0

    #input_df.loc[input_df[xy_dist] > ((input_df['r'] + np.sqrt(input_df['r']))/np.sqrt(input_df['r'])), det_class] = 0
    #input_df.loc[input_df[r_diff] > ((input_df['r'] + np.sqrt(input_df['r']))/np.sqrt(input_df['r'])), det_class] = 0
    #input_df.loc[input_df[xy_dist] > (input_df['r'] * 1.1), det_class] = 0
    #input_df.loc[input_df[r_diff] > (input_df['r'] * 1.05), det_class] = 0
    input_df.loc[input_df[xy_dist] > (input_df['r']/(pow(2, input_df['r']))), det_class] = 0
    input_df.loc[input_df[r_diff] > (input_df['r']/(pow(2, input_df['r']))), det_class] = 0
    input_df.loc[input_df[det_class] > 0, det_class] = 1
    input_df.loc[input_df[det_class] < 1, det_class] = 0
    print(input_df.gt_uid.value_counts())


    return input_df

def prepRFDataRF(input_df, drop_cols=['2_uid', '3_uid', '4_uid', '5_uid'], det_class='gt_uid'):
    """ This is a convenience function to clean up my predictions for binary
        classifiers.

    :param input_df: a dataframe with numerical UIDs.
    :type input_df: pd dataframe.
    :param drop_cols: Include any columns to be dropped in this list, defaults to ['2_uid', '3_uid', '4_uid', '5_uid']
    :type drop_cols: list of str
    :param det_class: column to use for tresholding, defaults to 'gt_uid'
    :type det_class: str


    :return: A cleaned up dataframe
    :rtype: pd dataframe
    """        

    input_df.drop(columns=drop_cols, inplace=True)
    input_df.fillna(1000, inplace=True)    
    input_df.loc[input_df[det_class] > 0, det_class] = 1
    input_df.loc[input_df[det_class] < 1, det_class] = 0

    return input_df

def binaryUpsample(input_df, uid='gt_uid'):
    #train = prepRFData(input_df)
    #print('upsample')

    ids = input_df[uid]
    #print(ids.value_counts)

    #count_class_0, count_class_1 = train.gt_uid.value_counts()
    count_class_0, count_class_1 = ids.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = input_df[input_df[uid] == 0]
    df_class_1 = input_df[input_df[uid] == 1]
    print(df_class_1.shape)
    df_class_1_over = resample(df_class_1, n_samples=count_class_0, replace=True)
    print(df_class_1_over.shape)

    upsampled = pd.concat([df_class_0, df_class_1_over], axis=0)    

    return upsampled

def trainSVC(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:11].values
    y = df_test_over.iloc[:,12].values

    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:11].values
    val_y = val.iloc[:,12].values

    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))
    clf.fit(X, y)

    y_pred = clf.predict(val_X)

    joblib.dump(clf, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def trainVoter(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:11].values
    y = df_test_over.iloc[:,12].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:11].values
    val_y = val.iloc[:,12].values

    rf = RandomForestClassifier(random_state=0, criterion='entropy')
    erf = ExtraTreesClassifier(random_state=0, criterion='entropy')
    ada = AdaBoostClassifier(random_state=0)
    grad = GradientBoostingClassifier(random_state=0)
    svc = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=5000))
    #svc = make_pipeline(StandardScaler(), SVC(random_state=0, tol=1e-5, probability=True))

    combo = VotingClassifier(estimators=[('erf', erf), ('ada', ada), ('grad', grad)], voting='soft')

    combo.fit(X, y)
    y_pred = combo.predict(val_X)

    joblib.dump(combo, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def trainStacker(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:11].values
    y = df_test_over.iloc[:,12].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:11].values
    val_y = val.iloc[:,12].values

    rf = RandomForestClassifier(random_state=0, criterion='entropy')
    erf = ExtraTreesClassifier(random_state=0, criterion='entropy')
    ada = AdaBoostClassifier(random_state=0)
    grad = GradientBoostingClassifier(random_state=0)
    svc = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual=False))

    combo = StackingClassifier(estimators=[('erf', erf), ('svc', svc), ('grad', grad)])

    combo.fit(X, y)
    y_pred = combo.predict(val_X)

    joblib.dump(combo, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)    


def trainRF(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:11].values
    y = df_test_over.iloc[:,12].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:11].values
    val_y = val.iloc[:,12].values

    regressor = RandomForestClassifier(random_state=0, criterion='entropy')
    regressor.fit(X, y)
    y_pred = regressor.predict(val_X)

    joblib.dump(regressor, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def trainERF(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:-3].values
    y = df_test_over.iloc[:,-3].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:-3].values
    val_y = val.iloc[:,-3].values

    regressor = ExtraTreesClassifier(random_state=0, criterion='entropy')
    regressor.fit(X, y)
    y_pred = regressor.predict(val_X)

    joblib.dump(regressor, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def searchERF(train_df):
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

    train = prepRFData(train_df)

    upsampled = binaryUpsample(train)

    X = upsampled.iloc[:,3:11].values
    y = upsampled.iloc[:,12].values

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    bootstrap = [True, False]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_features = ['auto', 'sqrt']

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion' : criterion,
                    'bootstrap': bootstrap}
    erf = ExtraTreesClassifier()
    erf_random = RandomizedSearchCV(estimator = erf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
    # Fit the random search model
    erf_random.fit(X, y)

    print(erf_random.best_params_)

def searchGrad(train_df):
    train = prepRFData(train_df)

    upsampled = binaryUpsample(train)

    X = upsampled.iloc[:,3:-3].values
    y = upsampled.iloc[:,-3].values


    n_estimators = [100, 200, 300]
    max_depth = [3, 5, 10]


    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth}

    grad = GradientBoostingClassifier()
    grad_rand = GridSearchCV(grad, random_grid, cv = 3, verbose=2, n_jobs = 4, pre_dispatch='2*n_jobs')
    print('grid search done')
    grad_rand.fit(X, y)
    print('grid fit done')

    print(grad_rand.best_params_)

    results = pd.DataFrame.from_dict(grad_rand.cv_results_)

    results.to_csv('/Users/theo/muh_results.csv', index=False)


def trainAdaBoost(train_df, val_df, rf_model):
    train = prepRFData(train_df)
    val = prepRFData(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:11].values
    y = df_test_over.iloc[:,12].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:11].values
    val_y = val.iloc[:,12].values

    regressor = AdaBoostClassifier(random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(val_X)

    joblib.dump(regressor, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def trainGradBoost(train_df, val_df, rf_model):
    train = prepRFDataRF(train_df)
    val = prepRFDataRF(val_df)

    count_class_0, count_class_1 = train.gt_uid.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = train[train['gt_uid'] == 0]
    df_class_1 = train[train['gt_uid'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)

    df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    X = df_test_over.iloc[:,3:-3].values
    y = df_test_over.iloc[:,-3].values



    print(X.shape)
    print(y.shape)

    val_X = val.iloc[:,3:-3].values
    val_y = val.iloc[:,-3].values

    regressor = GradientBoostingClassifier(random_state=0)
    regressor.fit(X, y)
    y_pred = regressor.predict(val_X)

    joblib.dump(regressor, rf_model)

    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def testRF(test_df, rf_model):

    test = prepRFData(test_df)

    X = test.iloc[:,3:-3].values
    y = test.iloc[:,-3].values

    regressor = joblib.load(rf_model)
    test['y_pred'] = regressor.predict(X)

    cols = list(test_df)
    cols = cols[3:-3]

    print(confusion_matrix(y,test['y_pred']))
    print(classification_report(y,test['y_pred']))
    print(accuracy_score(y, test['y_pred']))
    tn, fp, fn, tp = confusion_matrix(y,test['y_pred']).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)
    plot_roc_curve(regressor, X, y)
    plot_precision_recall_curve(regressor, X, y)
    plot_confusion_matrix(regressor, X, y)
    #plt.barh(cols, regressor.feature_importances_)


    test.to_csv('/Users/theo/data/hough_dataset/219_detections.csv', index=False)

    plt.show()



train_csv = '/Users/theo/data/hough_dataset/good_enough_train.csv'
val_csv = '/Users/theo/data/hough_dataset/good_enough_val.csv'
test_csv = '/Users/theo/data/hough_dataset/good_enough_test.csv'
rf_model = '/Users/theo/data/hough_dataset/grad.joblib'


train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

#searchERF(train_df)
#searchGrad(train_df)
#trainGradBoost(train_df, val_df, rf_model)
#trainVoter(train_df, val_df, rf_model)
testRF(train_df, rf_model)


#print(train_df.shape)
#up = binaryUpsample(train_df)
#print(up.shape)


'''
input_df.drop(columns=['3_uid', '4_uid'], inplace=True)
input_df.fillna(100, inplace=True)

input_df.loc[input_df['gt_xy_dist'] > 0.5, 'gt_uid'] = 0
input_df.loc[input_df['gt_r_diff'] > 0.5, 'gt_uid'] = 0
input_df.loc[input_df['gt_uid'] > 0, 'gt_uid'] = 1
input_df.loc[input_df['gt_uid'] < 1, 'gt_uid'] = 0

count_class_0, count_class_1 = input_df.gt_uid.value_counts()
df_class_0 = input_df[input_df['gt_uid'] == 0]
df_class_1 = input_df[input_df['gt_uid'] == 1]
df_class_0_over = df_class_0.sample(count_class_1, replace=False)
df_test_over = pd.concat([df_class_0_over, df_class_1], axis=0)


X = df_test_over.iloc[:,3:11].values
y = df_test_over.iloc[:,12].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = RandomForestClassifier(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

joblib.dump(regressor, rf_model)



cols = list(df_test_over)
cols = cols[3:11]

#regressor = joblib.load(rf_model)

#print(regressor.feature_importances_)

sorted_idx = regressor.feature_importances_.argsort()

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)


print(sorted_idx)
plt.barh(cols, regressor.feature_importances_)
#plt.barh(cols[sorted_idx], regressor.feature_importances_[sorted_idx])
plt.show()
'''