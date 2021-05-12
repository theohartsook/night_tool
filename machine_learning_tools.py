from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from matplotlib import pyplot as plt

import pandas as pd
import joblib

def prepRFData(input_df, drop_cols=['2_uid', '3_uid', '4_uid', '5_uid', 'uid'], det_class='gt_uid'):
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

def binaryResample(input_df, uid='gt_uid'):
    """ This is a hacky convenience function to resample binary classification
        data by upsampling the minority class a little bit and downsampling
        the majority class by a lot.
        
    :param input_df: a dataframe returned by prepRFData.
    :type input_df: pd dataframe.

    :param uid: column to use for resampling, defaults to 'gt_uid'
    :type uid: str
    :param target: class to target for resampling, defaults to 1
    :type target: int

    :return: A resampled dataframe
    :rtype: pd dataframe
    """

    ids = input_df[uid]
    count_class_0, count_class_1 = ids.value_counts()
    print(count_class_0, count_class_1)
    df_class_0 = input_df[input_df[uid] == 0]
    df_class_1 = input_df[input_df[uid] == 1]
    if count_class_0 > count_class_1:
        resample_target = int(((count_class_0/2)+count_class_1)/2)
        resample_0 = resample(df_class_0, n_samples = resample_target, replace=False)
        resample_1 = resample(df_class_1, n_samples = resample_target, replace=True)
    elif count_class_1 > count_class_0:
        resample_target = int(count_class_0+(count_class_1/2)/2)
        resample_0 = resample(df_class_0, n_samples = resample_target, replace=True)
        resample_1 = resample(df_class_1, n_samples = resample_target, replace=False)

    resampled = pd.concat([resample_0, resample_1], axis=0)
    ids = resampled[uid]
    count_class_0, count_class_1 = ids.value_counts()
    print(count_class_0, count_class_1)

    return resampled

def binaryUpsample(input_df, uid='gt_uid', target=1):
    """ This is a convenience function to upsample binary classification
        data from prepRFData.
        
    :param input_df: a dataframe returned by prepRFData.
    :type input_df: pd dataframe.

    :param uid: column to use for resampling, defaults to 'gt_uid'
    :type uid: str
    :param target: class to target for resampling, defaults to 1
    :type target: int

    :return: An upsampled dataframe
    :rtype: pd dataframe
    """

    ids = input_df[uid]
    count_class_0, count_class_1 = ids.value_counts()
    df_class_0 = input_df[input_df[uid] == 0]
    df_class_1 = input_df[input_df[uid] == 1]

    if target == 0:
        resampled = resample(df_class_0, n_samples=count_class_1, replace=True)
        upsampled = pd.concat([resampled, df_class_1], axis=0)            
    elif target == 1:
        resampled = resample(df_class_1, n_samples=count_class_0, replace=True)
        upsampled = pd.concat([df_class_0, resampled], axis=0)    

    return upsampled

def binaryDownsample(input_df, uid='gt_uid', target=0):
    """ This is a convenience function to downsample binary classification
        data from prepRFData.

    :param input_df: a dataframe returned by prepRFData.
    :type input_df: pd dataframe.
    :param uid: column to use for resampling, defaults to 'gt_uid'
    :type uid: str
    :param target: class to target for resampling, defaults to 0
    :type target: int

    :return: A downsampled dataframe
    :rtype: pd dataframe
    """

    ids = input_df[uid]
    count_class_0, count_class_1 = ids.value_counts()
    df_class_0 = input_df[input_df[uid] == 0]
    df_class_1 = input_df[input_df[uid] == 1]

    if target == 0:
        resampled = resample(df_class_0, n_samples=count_class_1, replace=False)
        downsampled = pd.concat([resampled, df_class_1], axis=0)            
    elif target == 1:
        resampled = resample(df_class_1, n_samples=count_class_0, replace=False)
        downsampled = pd.concat([df_class_0, resampled], axis=0)    

    return downsampled

def trainClassifier(classifier, train_df, test_df, model_output, output_csv=None, x_indices=[3,-3], y_indices=[-3]):
    """ This is a conveience function to set up and train a sklearn classifier.

    :param train_df: the training dataset
    :type train_df: pd dataframe
    :param train_df: the testing dataset
    :type train_df: pd dataframe
    :param model_output: the filepath to save the trained classifier as a .joblib
    :type model_output: str
    :param_type x_indices: the start and stop indices for the X data, defaults to 3 and -3 because that works for my data
    :type x_indices: [int, int]
    :param_type y_indices: the start and stop indices for the Y data, defaults to -3 because that works for my data
    :type y_indices: [int, int]    
    """
    X = train_df.iloc[:,x_indices[0]:x_indices[1]].values
    y = train_df.iloc[:,y_indices[0]].values

    test_X = test_df.iloc[:,x_indices[0]:x_indices[1]].values
    test_y = test_df.iloc[:,y_indices[0]].values
    
    classifier.fit(X, y)
    test_df['pred']= classifier.predict(test_X)

    if output_csv is not None:
        test_df.to_csv(output_csv, index=False)

    joblib.dump(classifier, model_output)

    printMetrics(test_y, test_df['pred'])

def blendedClassifier(blender, train_df, test_df, model_output, output_csv=None, x_indices=[3,-3], y_indices=[-3]):
    X = train_df.iloc[:,x_indices[0]:x_indices[1]].values
    y = train_df.iloc[:,y_indices[0]].values

    test_X = test_df.iloc[:,x_indices[0]:x_indices[1]].values
    test_y = test_df.iloc[:,y_indices[0]].values

    test_df['pred'] = blender.fit(X,y)

    if output_csv is not None:
        test_df.to_csv(output_csv, index=False)

    joblib.dump(blender, model_output)

    printMetrics(test_y, test_df['pred'])


def printMetrics(val_y, y_pred):
    """ A little convenience function to print accuracy metrics. 
    
    :param val_y: The observed data.
    :type val_y: pd dataframe
    :param y_pred: The predicted data from classifier.predict(X)
    :type y_pred: sparse matrix
    """


    print(confusion_matrix(val_y,y_pred))
    print(classification_report(val_y,y_pred))
    print(accuracy_score(val_y, y_pred))
    tn, fp, fn, tp = confusion_matrix(val_y,y_pred).ravel()
    print('tn', tn, '\nfp', fp, '\nfn', fn, '\ntp', tp)

def parameterSweep(X, y, classifier, param_grid, num_workers=4, results_csv=None, iterations=None):
    """ This is a wrapper for GridSearchCV and RandomixedSearchCV.

    :param X: the input samples
    :type X: pd dataframe
    :param y: the input labels
    :type y: pd dataframe
    :param classifier: A sklearn classifier such as GradientBoostingClassifier
    :type classifier:
    :param param_grid: A dictionary with the parameters to be searched
    :type param_grid: dict
    :param num_workers: Number of jobs for the search to run, defaults to 4
    :type num_workers: int
    :param results_csv: A filepath to save the results of all searches at, defaults to None
    :type results_csv: str
    :param iterations: Controls whether it is a GridSearchCV or RandomizedSearchCV. If an
        int is provided, RandomizedSearchCV will run with that many iterations. If it is None
        GridSearchCV will run instead. Defaults to None

    :return: A dictionary with the best parameters.
    :rtype: dict
    """

    if iterations is not None:
        search = GridSearchCV(classifier, param_grid, cv=3, verbose=1, n_jobs=num_workers, pre_dispatch=2*num_workers)
    else:
        search = RandomizedSearchCV(classifier, param_grid, cv=3, verbose=1, n_iter=iterations, n_jobs=num_workers, pre_dispatch=2*num_workers)
    search.fit(X, y)

    if results_csv is not None:
        results = pd.DataFrame.from_dict(search.cv_results_)
        results.to_csv(results_csv, index=False)
        return search.best_params_
    else:
        return search.best_params_

def testClassifier(test_df, rf_model, results_csv=None, x_indices=[3,-3], y_indices=[-3], pred_column = 'y_pred'):
    X = test_df.iloc[:,x_indices[0]:x_indices[1]].values
    y = test_df.iloc[:,y_indices[0]].values

    classifier = joblib.load(rf_model)
    test_df[pred_column] = classifier.predict(X)

    printMetrics(y, test_df[pred_column])

    cols = list(test_df)
    cols = cols[3:-4]

    plt.barh(cols, classifier.feature_importances_)

    plot_roc_curve(classifier, X, y)
    plot_precision_recall_curve(classifier, X, y)
    plot_confusion_matrix(classifier, X, y)

    if results_csv is not None:
        test_df.to_csv(results_csv, index=False)

    plt.show()

train_csv = '/Users/theo/data/hough_dataset/best_only_train.csv'
val_csv = '/Users/theo/data/hough_dataset/good_enough_val.csv'
test_csv = '/Users/theo/data/hough_dataset/good_enough_test.csv'
#rf_model = '/Users/theo/data/hough_dataset/binary_best.joblib'
rf_model = 'big_joke.joblib'
grad_sweep = '/Users/theo/data/hough_dataset/grad.joblib'
val_pred_csv = '/Users/theo/data/hough_dataset/predicted_val.csv'
test_pred_csv = '/Users/theo/data/hough_dataset/predicted_test.csv'

gradboost = GradientBoostingClassifier(random_state=0, learning_rate=0.3, max_depth=10, n_estimators=500)

erf = ExtraTreesClassifier(random_state=0, criterion='entropy')

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

train = prepRFData(train_df)
val = prepRFData(val_df)
test = prepRFData(test_df)


#train = binaryUpsample(train)

'''

train.loc[train['gt_uid'] == 0, train['gt_uid']] = '0'
train.loc[train['gt_uid'] == 1, train['gt_uid']] = '1'
train.loc[train['r'] > 0.2 & train['gt_uid'] == 1, train['gt_uid']] = '2'
val.loc[val['gt_uid'] == 0, val['gt_uid']] = '0'
val.loc[val['gt_uid'] == 1, val['gt_uid']] = '1'
val.loc[val['r'] > 0.2 & val['gt_uid'] == 1, val['gt_uid']] = '2'
'''

classifiers = [gradboost, erf]

#blender = StackingClassifier(estimators=[('erf', erf,),('grad', gradboost)])

#blendedClassifier(blender, train, val, rf_model, output_csv) 


#trainClassifier(gradboost, train, val, rf_model)

#testClassifier(test, grad_sweep)