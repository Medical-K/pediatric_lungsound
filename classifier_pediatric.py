import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from imblearn.over_sampling import SMOTE
from numpy import sqrt, argmax
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb
import json

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)
    
    inputs = np.array(data["mfcc"])
    name = np.array(data["name"])
    targets = np.array(data["labels"])    
    return inputs, name, targets

def label_to_num(targets, classes):
    class2index = {}
    for symptom in classes:
        if symptom not in class2index.keys():
            class2index[symptom]=len(class2index)
    output = []
    for tg in targets:
        output.append(class2index[tg])
    return output, class2index

def calculate_diagnostic_performance (actual_predicted):
    """ Calculate diagnostic performance.
    
    Takes a Numpy array of 1 and zero, two columns: actual and predicted
    
    Note that some statistics are repeats with different names
    (precision = positive_predictive_value and recall = sensitivity).
    Both names are returned
    
    Returns a dictionary of results:
        
    1) accuracy: proportion of test results that are correct    
    2) sensitivity: proportion of true +ve identified
    3) specificity: proportion of true -ve identified
    4) positive likelihood: increased probability of true +ve if test +ve
    5) negative likelihood: reduced probability of true +ve if test -ve
    6) false positive rate: proportion of false +ves in true -ve patients
    7) false negative rate:  proportion of false -ves in true +ve patients
    8) positive predictive value: chance of true +ve if test +ve
    9) negative predictive value: chance of true -ve if test -ve
    10) precision = positive predictive value 
    11) recall = sensitivity
    12) f1 = (2 * precision * recall) / (precision + recall)
    13) positive rate = rate of true +ve (not strictly a performance measure)
    """
    # Calculate results
    actual_positives = actual_predicted[:, 0] == 1
    actual_negatives = actual_predicted[:, 0] == 0
    test_positives = actual_predicted[:, 1] == 1
    test_negatives = actual_predicted[:, 1] == 0
    test_correct = actual_predicted[:, 0] == actual_predicted[:, 1]
    accuracy = np.average(test_correct)
    true_positives = actual_positives & test_positives
    true_negatives = actual_negatives & test_negatives
    sensitivity = np.sum(true_positives) / np.sum(actual_positives)
    specificity = np.sum(true_negatives) / np.sum(actual_negatives)
    positive_likelihood = sensitivity / (1 - specificity)
    negative_likelihood = (1 - sensitivity) / specificity
    false_positive_rate = 1 - specificity
    false_negative_rate = 1 - sensitivity
    positive_predictive_value = np.sum(true_positives) / np.sum(test_positives)
    negative_predictive_value = np.sum(true_negatives) / np.sum(test_negatives)
    precision = positive_predictive_value
    recall = sensitivity
    f1 = (2 * precision * recall) / (precision + recall)
    positive_rate = np.mean(actual_predicted[:,1])
    
    # Add results to dictionary
    performance = {}
    performance['accuracy'] = accuracy
    performance['sensitivity'] = sensitivity
    performance['specificity'] = specificity
    performance['positive_likelihood'] = positive_likelihood
    performance['negative_likelihood'] = negative_likelihood
    performance['false_positive_rate'] = false_positive_rate
    performance['false_negative_rate'] = false_negative_rate
    performance['positive_predictive_value'] = positive_predictive_value
    performance['negative_predictive_value'] = negative_predictive_value
    performance['precision'] = precision
    performance['recall'] = recall
    performance['f1'] = f1
    performance['positive_rate'] = positive_rate
    return performance

def normalise (X_train,X_test):
    """Normalise X data, so that training set has mean of zero and standard
    deviation of one"""
    # Initialise a new scaling object for normalising input data
    sc=StandardScaler() 
    # Set up the scaler just on the training set
    sc.fit(X_train)
    # Apply the scaler to the training and test sets
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)
    return X_train_std, X_test_std

def print_diagnostic_results (performance):
    """Iterate through, and print, the performance metrics dictionary"""
    print('\nMachine learning diagnostic performance measures:')
    print('-------------------------------------------------')
    for key, value in performance.items():
        print (key,'= %0.3f' %value) # print 3 decimal places
    return

def print_feaure_importances (model, features):
    df = pd.DataFrame()
    df['feature'] = features
    df['importance'] = model.feature_importances_
    #df = df.sort_values('importance', ascending = False)
    df = df.sort_values('feature', ascending = False)
    return df


##############################################################
def generate_data(TASK_TYPE, crackle_X, crackle_y, crackle_name,
                            wheezing_X, wheezing_y, wheezing_name,
                            normal_X, normal_y, normal_name):
    if TASK_TYPE == 'multi_class':
        print('** Task: 3-class classification (normal vs. wheezing vs. crackle)')
        print('---------------------------')
        print('<After flattening>')
        print('crackle : ', crackle_X.shape)
        print('wheezing: ', wheezing_X.shape)
        print('Normal  : ', normal_X.shape)
        print('---------------------------')
        data = np.vstack((crackle_X, wheezing_X, normal_X))
        target = np.hstack((crackle_y, wheezing_y, normal_y))
        name = np.hstack((crackle_name, wheezing_name, normal_name))
        classes = ['crackle', 'wheezing', 'normal']
        data_name_list = np.array(list(crackle_name) + list(wheezing_name) + list(normal_name))

        target, class2index = label_to_num(target, classes)
        target = np.array(target)
        X, y, name = shuffle(data, target, name)
        print('<Final data shape>')
        print('data : ', X.shape)
        print('target: ', y.shape)        
    elif TASK_TYPE == 'abnormal_binary':
        print('** Task: Normal vs. Abnormal (crackle + wheezing)')
        abnormal_X = np.vstack((crackle_X, wheezing_X))        
        abnormal_y = np.repeat('abnormal', len(crackle_y) + len(wheezing_y))
        abnormal_name = np.hstack((crackle_name, wheezing_name))
        print('---------------------------')
        print('<After flattening>')
        print('normal : ', normal_X.shape)
        print('abnormal: ', abnormal_X.shape)
        print('---------------------------')        

        data = np.vstack((normal_X, abnormal_X))
        target = np.hstack((normal_y, abnormal_y))
        name = np.hstack((normal_name, abnormal_name))
        classes = ['normal', 'abnormal']
        data_name_list = np.array(list(normal_name) + list(crackle_name) + list(wheezing_name))

        target, class2index = label_to_num(target, classes)
        target = np.array(target)
        idx = np.arange(len(data))
        X, y, name = shuffle(data, target, name)
        print('<Final data shape>')
        print('data : ', X.shape)
        print('target: ', y.shape)   

    elif TASK_TYPE == 'abnormal_vertical':
        print('** Task: crackle vs. wheezing')
        print('---------------------------')
        print('<After flattening>')
        print('crackle : ', crackle_X.shape)
        print('wheezing: ', wheezing_X.shape)
        print('---------------------------')
        data = np.vstack((crackle_X, wheezing_X))
        target = np.hstack((crackle_y, wheezing_y))
        name = np.hstack((crackle_name, wheezing_name))
        classes = ['crackle', 'wheezing']
        data_name_list = np.array(list(crackle_name) + list(wheezing_name))

        target, class2index = label_to_num(target, classes)
        target = np.array(target)
        idx = np.arange(len(data))
        X, y, name = shuffle(data, target, name)
        print('<Final data shape>')
        print('data : ', X.shape)
        print('target: ', y.shape) 

    elif TASK_TYPE == 'normal_crackle':
        print('** Task: Normal vs. Crackle')
        print('---------------------------')
        print('<After flattening>')
        print('normal : ', normal_X.shape)
        print('crackle: ', crackle_X.shape)
        print('---------------------------')
        data = np.vstack((normal_X, crackle_X))
        target = np.hstack((normal_y, crackle_y))
        name = np.hstack((normal_name, crackle_name))
        classes = ['normal', 'crackle']
        data_name_list = np.array(list(normal_name) + list(crackle_name))

        target, class2index = label_to_num(target, classes)
        target = np.array(target)
        idx = np.arange(len(data))
        X, y, name = shuffle(data, target, name)
        print('<Final data shape>')
        print('data : ', X.shape)
        print('target: ', y.shape)  
    elif TASK_TYPE == 'normal_wheezing':
        print('** Task: Normal vs. Wheezing')
        print('---------------------------')
        print('<After flattening>')
        print('normal : ', normal_X.shape)
        print('Wheezing: ', wheezing_X.shape)
        print('---------------------------')
        data = np.vstack((normal_X, wheezing_X))
        target = np.hstack((normal_y, wheezing_y))
        name = np.hstack((normal_name, wheezing_name))
        classes = ['normal', 'wheezing']
        data_name_list = np.array(list(normal_name) + list(wheezing_name))

        target, class2index = label_to_num(target, classes)
        target = np.array(target)
        idx = np.arange(len(data))
        X, y, name = shuffle(data, target, name)
        print('<Final data shape>')
        print('data : ', X.shape)
        print('target: ', y.shape)                 
    return X, y, name, classes, data_name_list

from sklearn.svm import SVC
def SVM_model(X, y, classes, c, EXTERNAL_VALIDATION, X_external, y_external, name_external, SMOTE_METHOD):
    cmap = 'magma'
    print('SVM model prediction!')
    print("SMOTE METHOD -------------------- ", SMOTE_METHOD)
    y_true_stack, y_pred_stack, y_prob_stack = [], [], []
    y_true_ext_stack, y_pred_ext_stack = [], []
    accuracy_stack = []
    model_list = []
    BEST_EXT_ACC = 0
    n_splits = 10    
    skf = StratifiedKFold(n_splits = n_splits)
    for train_index, test_index in skf.split(X, y):        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ## Normalization
        #scaler = StandardScaler()   # MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        #X_external = scaler.transform(X_external.reshape(-1, X_external.shape[-1])).reshape(X_external.shape)
        if SMOTE_METHOD == True:
            smote = SMOTE(random_state=0)
            X_train,y_train = smote.fit_resample(X_train,y_train)              
        
        # Train model
        clf = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=c, probability=True)
        model = clf.fit(X_train, y_train)
        model_list.append(model)

        # Produce results for test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_test)
        
        y_true_stack.extend(y_test)
        y_pred_stack.extend(y_pred) 
        y_prob_stack.extend(y_prob)           
        accuracy_stack.append(accuracy)
        print("train: {}, test: {}, acc={:.4f}".format(len(train_index), len(test_index), accuracy))    

    if EXTERNAL_VALIDATION:            
        ###################################
        # for external validation setting    
        print('*External validation start!>')    
        scaler = StandardScaler()
        X_external = scaler.fit_transform(X_external.reshape(-1, X_external.shape[-1])).reshape(X_external.shape)
        y_ext_true = y_external

        y_ext_pred_all = []
        for mo in model_list:
            y_ext_pred_all.append(mo.predict(X_external))        
        y_ext_pred_all = np.array(y_ext_pred_all)

        y_ext_pred = []
        for case in range(y_ext_pred_all.shape[1]):
            y_ext_pred.append(np.argmax(np.bincount(y_ext_pred_all[:, case])))
        
        BEST_EXT_ACC = np.sum(np.array(y_ext_true)==np.array(y_ext_pred)) / len(y_ext_true)
        print('<Best model found> ACC for external validation: ', BEST_EXT_ACC)
        np.save('ML_result/SVM_pred_stack_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), y_ext_pred)
        np.save('ML_result/SVM_true_stack_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), y_ext_true)
        np.save('ML_result/SVM_name_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), name_external)                


    print('************ Final average *************')
    final_accuracy = np.sum(np.array(y_true_stack)==np.array(y_pred_stack)) / len(y_true_stack)
    final_avgaccuracy = np.average(accuracy_stack)
    print('ACC: ', final_accuracy)
    print('Avg ACC: ', final_avgaccuracy)
    final_acc = np.around(np.max((final_accuracy, final_avgaccuracy)), 3)
    print('---------------------------------')
    print('External ACC average: ', BEST_EXT_ACC)
    print('*****************************************')

    # Confusion matrix      
    kfold_results = confusion_matrix(y_true_stack, y_pred_stack, labels=list(range(len(classes))))
    df_cm = pd.DataFrame(kfold_results, index = classes,
                    columns = classes)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Oranges', annot_kws={'size':25})
    ax.set_xlabel("Predicted", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)
    ax.set_ylabel("Actual", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)   
    plt.savefig('ML_result/SVM_cm_{}_{}.png'.format(final_acc, TASK_TYPE))
    np.save('ML_result/SVM_pred_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_pred_stack)
    np.save('ML_result/SVM_prob_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_prob_stack)
    np.save('ML_result/SVM_true_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_true_stack)
    print('Before external - Save finish!')
    print('##########################################################################')

def GP_model(X, y, classes, SMOTE_METHOD):
    cmap = 'magma'
    print('Gaussian Process model prediction!')
    print("SMOTE METHOD -------------------- ", SMOTE_METHOD)
    y_true_stack, y_pred_stack, y_prob_stack = [], [], []
    accuracy_stack = []
    n_splits = 10
    skf = StratifiedKFold(n_splits = n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ## Normalization
        #scaler = StandardScaler()   # MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        if SMOTE_METHOD == True:            
            smote = SMOTE(random_state=0)
            X_train,y_train = smote.fit_resample(X_train,y_train)          
        
        # Train model
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        model = clf.fit(X_train, y_train)

        # Produce results for test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_test)
        
        y_true_stack.extend(y_test)
        y_pred_stack.extend(y_pred) 
        y_prob_stack.extend(y_prob)   
        accuracy_stack.append(accuracy)
        print("train: {}, test: {}, acc={:.4f}".format(len(train_index), len(test_index), accuracy))    
        
    print('************ Final average *************')
    final_accuracy = np.sum(np.array(y_true_stack)==np.array(y_pred_stack)) / len(y_true_stack)
    final_avgaccuracy = np.average(accuracy_stack)
    print('ACC: ', final_accuracy)
    print('Avg ACC: ', final_avgaccuracy)
    final_acc = np.around(np.max((final_accuracy, final_avgaccuracy)), 3)
    print('*****************************************')

    # Confusion matrix      
    kfold_results = confusion_matrix(y_true_stack, y_pred_stack, labels=list(range(len(classes))))
    df_cm = pd.DataFrame(kfold_results, index = classes,
                    columns = classes)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Oranges', annot_kws={'size':25})
    ax.set_xlabel("Predicted", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)
    ax.set_ylabel("Actual", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)   
    plt.savefig('ML_result/GP_cm_{}_{}.png'.format(final_acc, TASK_TYPE))
    np.save('ML_result/GP_pred_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_pred_stack)
    np.save('ML_result/GP_prob_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_prob_stack)
    np.save('ML_result/GP_true_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_true_stack)
    print('Save finish!')    

def Neunet_model(X, y, classes, SMOTE_METHOD):
    cmap = 'magma'
    print('Neural Network model prediction!')
    print("SMOTE METHOD -------------------- ", SMOTE_METHOD)
    y_true_stack, y_pred_stack, y_prob_stack = [], [], []
    accuracy_stack = []
    n_splits = 10
    skf = StratifiedKFold(n_splits = n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ## Normalization
        #scaler = StandardScaler()   # MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        if SMOTE_METHOD == True:            
            smote = SMOTE(random_state=0)
            X_train,y_train = smote.fit_resample(X_train,y_train)          
        
        # Train model
        clf = MLPClassifier(alpha=1, max_iter=1000)
        model = clf.fit(X_train, y_train)

        # Produce results for test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_test)
        
        y_true_stack.extend(y_test)
        y_pred_stack.extend(y_pred) 
        y_prob_stack.extend(y_prob)   
        accuracy_stack.append(accuracy)
        print("train: {}, test: {}, acc={:.4f}".format(len(train_index), len(test_index), accuracy))    
        
    print('************ Final average *************')
    final_accuracy = np.sum(np.array(y_true_stack)==np.array(y_pred_stack)) / len(y_true_stack)
    final_avgaccuracy = np.average(accuracy_stack)
    print('ACC: ', final_accuracy)
    print('Avg ACC: ', final_avgaccuracy)
    final_acc = np.around(np.max((final_accuracy, final_avgaccuracy)), 3)
    print('*****************************************')

    # Confusion matrix      
    kfold_results = confusion_matrix(y_true_stack, y_pred_stack, labels=list(range(len(classes))))
    df_cm = pd.DataFrame(kfold_results, index = classes,
                    columns = classes)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Oranges', annot_kws={'size':25})
    ax.set_xlabel("Predicted", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)
    ax.set_ylabel("Actual", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)   
    plt.savefig('ML_result/Neunet_cm_{}_{}.png'.format(final_acc, TASK_TYPE))
    np.save('ML_result/Neunet_pred_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_pred_stack)
    np.save('ML_result/Neunet_prob_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_prob_stack)
    np.save('ML_result/Neunet_true_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_true_stack)
    print('Save finish!')

def RF_model(X, y, classes, SMOTE_METHOD):
    cmap = 'magma'
    print('Random Forest model prediction!')
    print("SMOTE METHOD -------------------- ", SMOTE_METHOD)
    y_true_stack, y_pred_stack, y_prob_stack = [], [], []
    accuracy_stack = []
    n_splits = 10
    skf = StratifiedKFold(n_splits = n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ## Normalization
        #scaler = StandardScaler()   # MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        if SMOTE_METHOD == True:            
            smote = SMOTE(random_state=0)
            X_train,y_train = smote.fit_resample(X_train,y_train)          
        
        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=CPU_THREADS)
        model = clf.fit(X_train, y_train)

        # Produce results for test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        accuracy = np.sum(np.equal(y_test, y_pred)) / len(y_test)
        
        y_true_stack.extend(y_test)
        y_pred_stack.extend(y_pred) 
        y_prob_stack.extend(y_prob)   
        accuracy_stack.append(accuracy)
        print("train: {}, test: {}, acc={:.4f}".format(len(train_index), len(test_index), accuracy))    
        
    print('************ Final average *************')
    final_accuracy = np.sum(np.array(y_true_stack)==np.array(y_pred_stack)) / len(y_true_stack)
    final_avgaccuracy = np.average(accuracy_stack)
    print('ACC: ', final_accuracy)
    print('Avg ACC: ', final_avgaccuracy)
    final_acc = np.around(np.max((final_accuracy, final_avgaccuracy)), 3)
    print('*****************************************')

    # Confusion matrix      
    kfold_results = confusion_matrix(y_true_stack, y_pred_stack, labels=list(range(len(classes))))
    df_cm = pd.DataFrame(kfold_results, index = classes,
                    columns = classes)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Oranges', annot_kws={'size':25})
    ax.set_xlabel("Predicted", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)
    ax.set_ylabel("Actual", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)   
    plt.savefig('ML_result/RF_cm_{}_{}.png'.format(final_acc, TASK_TYPE))
    np.save('ML_result/RF_pred_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_pred_stack)
    np.save('ML_result/RF_prob_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_prob_stack)
    np.save('ML_result/RF_true_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_true_stack)
    print('Save finish!')    

from sklearn.ensemble import VotingClassifier
def Ensemble_model(X, y, name, classes, model_num, THRESHOLD, EXTERNAL_VALIDATION, X_external, y_external, name_external, SMOTE_METHOD):
    print('Ensemble model prediction!')
    print("SMOTE METHOD -------------------- ", SMOTE_METHOD)
    y_true_stack, y_pred_stack = [], []
    y_true_ext_stack, y_pred_ext_stack = [], []
    BEST_EXT_ACC = 0
    n_splits = 10
    final_suspect_list = []
    model_en_list = []
    skf = StratifiedKFold(n_splits = n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        name_train, name_test = name[train_index], name[test_index]
        accuracy_stack = []

        ## Normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape) 
        if EXTERNAL_VALIDATION:   
            X_external = scaler.transform(X_external.reshape(-1, X_external.shape[-1])).reshape(X_external.shape)
        if SMOTE_METHOD == True:
            smote = SMOTE(random_state=0)
            X_train,y_train = smote.fit_resample(X_train,y_train)              

        # Train model
        #model_1 = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        #model_2 = XGBClassifier(n_jobs=-1, random_state=0, reg_alpha=0.9, reg_lambda=0.2)    
        if model_num == 4:
            model_1 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1)
            model_2 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.5)
            model_3 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.5)
            model_4 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.8)
            model_list = [model_1, model_2, model_3, model_4]
            final_model = VotingClassifier(
                estimators=[('svc1', model_1), ('svc2', model_2), ('svc3', model_3), ('svc4', model_4)], voting='hard')    
            # training all the model on the train dataset
            final_model.fit(X_train, y_train)
            model_1.fit(X_train, y_train)
            model_2.fit(X_train, y_train)
            model_3.fit(X_train, y_train)
            model_4.fit(X_train, y_train)
            
            # Produce results for test set
            y_pred = final_model.predict(X_test)
            y_true_stack.extend(y_test)
            y_pred_stack.extend(y_pred)        
            acc_ensemble = np.sum(np.array(y_test)==np.array(y_pred)) / len(y_test)
            acc_model1 = np.sum(np.array(y_test)==np.array(model_1.predict(X_test))) / len(y_test)
            acc_model2 = np.sum(np.array(y_test)==np.array(model_2.predict(X_test))) / len(y_test)
            acc_model3 = np.sum(np.array(y_test)==np.array(model_3.predict(X_test))) / len(y_test)  
            acc_model4 = np.sum(np.array(y_test)==np.array(model_4.predict(X_test))) / len(y_test)
            accuracy_stack.append(acc_model1)
            accuracy_stack.append(acc_model2)
            accuracy_stack.append(acc_model3)
            accuracy_stack.append(acc_model4)
            print("train: {}, test: {}, acc= en: {:.4f} / 1:{:.4f} / 2:{:.4f} / 3: {:.4f} / 4: {:.4f}".format(len(train_index), len(test_index), acc_ensemble, acc_model1, acc_model2, acc_model3, acc_model4))
            # Find suspect list
            if np.abs(np.max(accuracy_stack) - np.min(accuracy_stack)) > THRESHOLD:
                suspect_list = judge_suspectlist(model_list, accuracy_stack, X_test, y_test, name_test)
                final_suspect_list.extend(suspect_list)
                print('Suspect list detected - {} found!'.format(len(suspect_list)))

        elif model_num == 5:
            model_1 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1)
            model_2 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.5)
            model_3 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.5)
            model_4 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.8)
            model_5 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.2)
            model_list = [model_1, model_2, model_3, model_4, model_5]            
            final_model = VotingClassifier(
                estimators=[('svc1', model_1), ('svc2', model_2), ('svc3', model_3), ('svc4', model_4), ('svc5', model_5)], voting='hard')    
            # training all the model on the train dataset
            final_model.fit(X_train, y_train)
            model_1.fit(X_train, y_train)
            model_2.fit(X_train, y_train)
            model_3.fit(X_train, y_train)
            model_4.fit(X_train, y_train)
            model_5.fit(X_train, y_train)            
            
            # Produce results for test set
            y_pred = final_model.predict(X_test)
            y_true_stack.extend(y_test)
            y_pred_stack.extend(y_pred)        
            acc_ensemble = np.sum(np.array(y_test)==np.array(y_pred)) / len(y_test)
            acc_model1 = np.sum(np.array(y_test)==np.array(model_1.predict(X_test))) / len(y_test)
            acc_model2 = np.sum(np.array(y_test)==np.array(model_2.predict(X_test))) / len(y_test)
            acc_model3 = np.sum(np.array(y_test)==np.array(model_3.predict(X_test))) / len(y_test)  
            acc_model4 = np.sum(np.array(y_test)==np.array(model_4.predict(X_test))) / len(y_test)
            acc_model5 = np.sum(np.array(y_test)==np.array(model_5.predict(X_test))) / len(y_test)
            accuracy_stack.append(acc_model1)
            accuracy_stack.append(acc_model2)
            accuracy_stack.append(acc_model3)
            accuracy_stack.append(acc_model4)
            accuracy_stack.append(acc_model5)
            print("train:{},test:{},acc=en: {:.4f} /1:{:.4f} /2:{:.4f} /3:{:.4f} /4:{:.4f} /5:{:.4f}".format(len(train_index), len(test_index), acc_ensemble, acc_model1, acc_model2, acc_model3, acc_model4, acc_model5))
            # Find suspect list
            if np.abs(np.max(accuracy_stack) - np.min(accuracy_stack)) > THRESHOLD:
                suspect_list = judge_suspectlist(model_list, accuracy_stack, X_test, y_test, name_test)
                final_suspect_list.extend(suspect_list)
                print('Suspect list detected - {} found!'.format(len(suspect_list)))

        elif model_num == 10:
            model_1 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1)
            model_2 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.2)
            model_3 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.4)
            model_4 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.6)
            model_5 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=1.8)
            model_6 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.8)
            model_7 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.6)
            model_8 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.4)
            model_9 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=0.2)
            model_10 = SVC(gamma='auto', class_weight='balanced', kernel='rbf', C=2)
            model_list = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]
            final_model = VotingClassifier(
                estimators=[('svc1', model_1), ('svc2', model_2), ('svc3', model_3), 
                            ('svc4', model_4), ('svc5', model_5), ('svc6', model_6), 
                            ('svc7', model_7), ('svc8', model_8), ('svc9', model_9), 
                            ('svc10', model_10)], voting='hard')    
            # training all the model on the train dataset
            final_model.fit(X_train, y_train)
            model_1.fit(X_train, y_train)
            model_2.fit(X_train, y_train)
            model_3.fit(X_train, y_train)
            model_4.fit(X_train, y_train)
            model_5.fit(X_train, y_train)   
            model_6.fit(X_train, y_train)
            model_7.fit(X_train, y_train)
            model_8.fit(X_train, y_train)
            model_9.fit(X_train, y_train)
            model_10.fit(X_train, y_train)                           
            
            # Produce results for test set
            y_pred = final_model.predict(X_test)
            y_true_stack.extend(y_test)
            y_pred_stack.extend(y_pred)        
            acc_ensemble = np.sum(np.array(y_test)==np.array(y_pred)) / len(y_test)
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_1.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_2.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_3.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_4.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_5.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_6.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_7.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_8.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_9.predict(X_test))) / len(y_test))
            accuracy_stack.append(np.sum(np.array(y_test)==np.array(model_10.predict(X_test))) / len(y_test))
            print("train:{},test:{},acc=en: {:.4f}".format(len(train_index), len(test_index), acc_ensemble))
            # Find suspect list
            if np.abs(np.max(accuracy_stack) - np.min(accuracy_stack)) > THRESHOLD:
                suspect_list = judge_suspectlist(model_list, accuracy_stack, X_test, y_test, name_test)
                final_suspect_list.extend(suspect_list)
                print('Suspect list detected - {} found!'.format(len(suspect_list)))
        
        model_en_list.append(final_model)
    if EXTERNAL_VALIDATION:            
        ###################################
        # for external validation setting    
        print('*External validation start!>')    
        scaler = StandardScaler()
        X_external = scaler.fit_transform(X_external.reshape(-1, X_external.shape[-1])).reshape(X_external.shape)
        y_ext_true = y_external

        y_ext_pred_all = []
        for mo in model_en_list:
            y_ext_pred_all.append(mo.predict(X_external))        
        y_ext_pred_all = np.array(y_ext_pred_all)

        y_ext_pred = []
        for case in range(y_ext_pred_all.shape[1]):
            y_ext_pred.append(np.argmax(np.bincount(y_ext_pred_all[:, case])))
        
        BEST_EXT_ACC = np.sum(np.array(y_ext_true)==np.array(y_ext_pred)) / len(y_ext_true)
        print('<Best model found> ACC for external validation: ', BEST_EXT_ACC)
        np.save('ML_result/0110_en/EN_pred_stack_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), y_ext_pred)
        np.save('ML_result/0110_en/EN_true_stack_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), y_ext_true)
        np.save('ML_result/0110_en/EN_name_{:.3f}_{}_ext.npy'.format(BEST_EXT_ACC, TASK_TYPE), name_external)                


    print('************ Final average *************')
    final_accuracy = np.sum(np.array(y_true_stack)==np.array(y_pred_stack)) / len(y_true_stack)
    final_avgaccuracy = np.average(accuracy_stack)
    print('ACC: ', final_accuracy)
    print('Avg ACC: ', final_avgaccuracy)
    final_acc = np.around(np.max((final_accuracy, final_avgaccuracy)), 3)
    print('---------------------------------')
    print('External ACC average: ', BEST_EXT_ACC)
    print('*****************************************')

    # Confusion matrix      
    kfold_results = confusion_matrix(y_true_stack, y_pred_stack, labels=list(range(len(classes))))
    df_cm = pd.DataFrame(kfold_results, index = classes,
                    columns = classes)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Oranges', annot_kws={'size':25})
    ax.set_xlabel("Predicted", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)
    ax.set_ylabel("Actual", fontsize = 15, fontdict=dict(weight='bold'), labelpad=10)   
    plt.savefig('ML_result/ensemble_cm_{}_{}.png'.format(final_acc, TASK_TYPE))
    np.save('ML_result/ensemble_pred_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_pred_stack)
    np.save('ML_result/ensemble_true_stack_{}_{}.npy'.format(final_acc, TASK_TYPE), y_true_stack)
    np.save('ML_result/ensemble{}_{}_suspectlist.npy'.format(model_num, TASK_TYPE), np.array(final_suspect_list))
    ## npy to csv
    #suspect_npy_to_csv(final_suspect_list, model_num, TASK_TYPE)
    print('Before external - Save finish!')
    
    print('##########################################################################')

def find_what_we_want(data_name_list, idx_list):
    output = []
    for idx in idx_list:
        output.append(data_name_list[idx])
    return np.array(output)

def judge_suspectlist(model_list, accuracy_stack, X_test, y_test, name_test):    
    best_model = model_list[np.argmax(accuracy_stack)]
    worst_model = model_list[np.argmin(accuracy_stack)]
    best_result = np.array(y_test)==np.array(best_model.predict(X_test))
    worst_result = np.array(y_test)==np.array(worst_model.predict(X_test))
    suspect_list = []
    for idx in range(len(best_result)):
        ## Criteria: best_result=True & worst_result=False
        if (best_result[idx] == True) and (worst_result[idx] == False):
            suspect_list.append(name_test[idx])
    return suspect_list   
    
def suspect_npy_to_csv(suspect_list, model_num, TASK_TYPE):
    df = pd.read_csv('suspect_detection.csv')[['date', 'file', 'lung']]
    final_dic = {}
    for name in suspect_list:
        final_dic[name] = df['lung'][df['date']==name.split('-')[0]][df['file']==int(name.split('-')[-1])].item()
    final_df = pd.DataFrame.from_dict(final_dic, orient='index', columns=['label'])    
    final_df.to_csv('ensemble{}_{}_suspectlist.csv'.format(model_num, TASK_TYPE))    


###########################################
CPU_THREADS = 80
TASK_TYPE = 'abnormal_binary'   # multi_class, abnormal_binary, abnormal_vertical, normal_crackle, normal_wheezing
MODEL_TYPE = 'GP'                    # SVM, EN, RF, GP
model_num = 10
THRESHOLD = 0.04                     
c = 1.5
SMOTE_METHOD = True
EXTERNAL_VALIDATION = False

max_epochs = 50
random_seed = 100
###########################################
# Total data load
print('**Data loaded..')
crackle_X, crackle_name, crackle_y = load_data('Preprocessed_data_icbhi/thinning_crackle_MFCC_ICBHI_pediatric.json')
print('   - Crackle data load finished')
wheezing_X, wheezing_name, wheezing_y = load_data('Preprocessed_data_icbhi/thinning_wheezing_MFCC_ICBHI_pediatric.json')
print('   - Wheezing data load finished')
normal_X, normal_name, normal_y = load_data('Preprocessed_data_icbhi/thinning_normal_MFCC_ICBHI_pediatric.json')
print('   - Normal data load finished')

print('---------------------------')
print('crackle : ', crackle_X.shape)
print('wheezing: ', wheezing_X.shape)
print('Normal  : ', normal_X.shape)
print('---------------------------')
crackle_X = np.reshape(crackle_X, (crackle_X.shape[0], -1))
wheezing_X = np.reshape(wheezing_X, (wheezing_X.shape[0], -1))
normal_X = np.reshape(normal_X, (normal_X.shape[0], -1))
###########################################
## make the result directory
os.makedirs('ML_result', exist_ok=True)
## generate data
X, y, name, classes, data_name_list = generate_data(TASK_TYPE, crackle_X, crackle_y, crackle_name,
                                                wheezing_X, wheezing_y, wheezing_name,
                                                normal_X, normal_y, normal_name)

## apply model
if MODEL_TYPE == 'RF':
    RF_model(X, y, classes, SMOTE_METHOD)
if MODEL_TYPE == 'GP':
    GP_model(X, y, classes, SMOTE_METHOD)    
elif MODEL_TYPE == 'SVM':
    if not EXTERNAL_VALIDATION:
        X_external = 'dum'
        y_external = 'dum'
        name_external = 'dum'
    SVM_model(X, y, classes, c, EXTERNAL_VALIDATION, X_external, y_external, name_external, SMOTE_METHOD)
elif MODEL_TYPE == 'EN':
    if not EXTERNAL_VALIDATION:
        X_external = 'dum'
        y_external = 'dum'    
        name_external = 'dum'
    Ensemble_model(X, y, name, classes, model_num, THRESHOLD, EXTERNAL_VALIDATION, X_external, y_external, name_external, SMOTE_METHOD)