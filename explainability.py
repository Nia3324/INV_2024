import numpy as np
import pandas as pd
import shap
import copy

from difflib import get_close_matches

import tensorflow as tf
import torch.nn as nn


def algorithm_class_score(clf, verbose = False):
    """
    Explainability score based on the model class. More complex models, will have a lower score.
        :param clf: the classifier 
        :return: normalized score of [1, 5]
    """
    # Based on literature research and qualitative analysis of each learning technique. For more information see gh-pages/explainability/taxonomy
    alg_score = {
    "RandomForestClassifier": 4,
    "KNeighborsClassifier": 3,
    "SVC": 2,
    "GaussianProcessClassifier": 3,
    "DecisionTreeClassifier": 5,
    "MLPClassifier": 1,
    "AdaBoostClassifier": 3,
    "GaussianNB": 3.5,
    "QuadraticDiscriminantAnalysis": 3,
    "LogisticRegression": 4,
    "LinearRegression": 3.5,
    }

    clf_name = type(clf).__name__
    if verbose: print(clf_name)

    # Check if the clf_name is in the dictionary
    if clf_name in alg_score:
        exp_score = alg_score[clf_name]
        return exp_score 

    # Check if the model is a Neural Network
    if isinstance(clf, tf.keras.Model) or isinstance(clf, tf.Module) or isinstance(clf, nn.Module):
        return 1
    
    # If not, try to find a close match
    close_matches = get_close_matches(clf_name, alg_score.keys(), n=1, cutoff=0.6)
    if close_matches:
        exp_score = alg_score[close_matches[0]]
        return exp_score
    
    # If no close match found 
    print(f"No matching score found for '{clf_name}'")
    return None


def correlated_features_score(dataset, thresholds=[0.05, 0.16, 0.28, 0.4], target_column=None, verbose=False):
    """
    Correlaation score from test and train data. The higher the score, the smaller the percentage of features with high 
    coorelation in relation to the average coorelation.
        :param dataset: testing dataset
        :param thresholds: bin thresholds for normalization
        :param target_column: if not None, target column will be excluded
        :param verbose: show features with high coorelation
        :return: normalized score of [1, 5]
    """

    if type(dataset) != 'pandas.core.frame.DataFrame':
        dataset = pd.DataFrame(dataset)

    dataset = copy.deepcopy(dataset)
     
    if target_column:
        X_test = dataset.drop(target_column, axis=1)
    else:
        X_test = dataset.iloc[:,:-1]

    X_test = X_test._get_numeric_data()
    corr_matrix =X_test.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Compute average and standar deviation from upper correlation matrix 
    avg_corr = upper.values[np.triu_indices_from(upper.values,1)].mean()
    std_corr = upper.values[np.triu_indices_from(upper.values,1)].std()

    # Find features with correlation greater than avg_corr + std_corr
    to_drop = [column for column in upper.columns if any(upper[column] > (avg_corr + std_corr))]
    if verbose: print(f"Removed features: {to_drop}")
    
    pct_drop = len(to_drop)/len(X_test.columns)
    #score = 5-np.digitize(pct_drop, thresholds, right=True) 
    
    return (1 - pct_drop)


def model_size_score(dataset, thresholds = [10,30,100,500]):
    # add comment
    # if NN then return n_parameters
    # else
    dist_score = 5- np.digitize(dataset.shape[1]-1 , thresholds, right=True) # -1 for the id?
    return dist_score


def feature_importance_score(clf, thresholds = [0.05, 0.1, 0.2, 0.3]):
    """
    Percentage of features that concentrates the majority of all importance.
        :param clf: the classifier 
        :param thresholds: bin thresholds for normalization
        :return: normalized score of [1, 5]
    """

    distri_threshold = 0.5

    regression = ['LogisticRegression', 'LogisticRegression']
    classifier = ['RandomForestClassifier', 'DecisionTreeClassifier']

    if (type(clf).__name__ in regression) or (get_close_matches(type(clf).__name__, regression, n=1, cutoff=0.6)): 
        importance = clf.coef_.flatten()

        total = 0
        for i in range(len(importance)):
            total += abs(importance[i])

        for i in range(len(importance)):
            importance[i] = abs(importance[i]) / total

    elif  (type(clf).__name__ in classifier) or (get_close_matches(type(clf).__name__, classifier, n=1, cutoff=0.6)):
        importance = clf.feature_importances_
   
    else:
        return None

    # absolut values
    importance = importance
    indices = np.argsort(importance)[::-1] # indice of the biggest value in the importance list
    importance = importance[indices]
    
    # percentage of features that concentrate distri_threshold percent of all importance
    pct_dist = sum(np.cumsum(importance) < distri_threshold) / len(importance)
    #score = np.digitize(pct_dist, thresholds, right=False) + 1 
    
    return pct_dist


def predict(model):
    return model.predict_proba if hasattr(model, 'predict_proba') else model.predict


def cv_shap_score(clf, dataset):
    """
    Coefficient of variance from the shap values of the features.
        :param clf: the classifier
        :param dataset: testing data

    """
     
    # Create a background dataset (subset of your training data)
    background = shap.sample(dataset, 100)  # Using 100 samples for background

    # Initialize KernelExplainer with the prediction function and background dataset
    explainer = shap.KernelExplainer(predict(clf), background)

    # Calculate SHAP values for the dataset you want to explain
    shap_values = explainer.shap_values(dataset.iloc[0])
  
    # calculare variance of absolute shap values 
    if shap_values is not None and len(shap_values) > 0:
        sums = np.array([abs(shap_values[i]).sum() for i in range(len(shap_values))])
        cv = np.std(sums) / np.mean(sums)
        return cv
    
    else: return None
