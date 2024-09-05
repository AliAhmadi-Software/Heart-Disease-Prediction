import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.metrics import confusion_matrix


def k_fold(x_fold, y_fold,algo,n_splits=10,shuffle=True,random_state=1,average='binary'):
    precision = []
    recall = []
    F1 = []
    accuracy = []
    specificity=[]

# y_true = [0, 0, 0, 1, 1, 1, 1, 1]
# y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# specificity = tn / (tn+fp)
    cv = KFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    for train_index, test_index in cv.split(x_fold):
        # print("Train Index: ", train_index, "\n")
        # print("Test Index: ", test_index)

        X_train, X_test, Y_train, Y_test = x_fold[train_index], x_fold[test_index], y_fold[train_index], y_fold[test_index]
        algo.fit(X_train, Y_train)
        precision.append(precision_score(Y_test,algo.predict(X_test),average=average))
        recall.append(recall_score(Y_test,algo.predict(X_test),average=average))
        F1.append(f1_score(Y_test,algo.predict(X_test),average=average))
        accuracy.append(algo.score(X_test,Y_test))

        tn, fp, fn, tp = confusion_matrix(Y_test,algo.predict(X_test)).ravel()
        specificity.append(tn / (tn+fp))

    return np.array([np.array(accuracy).mean(),np.array(recall).mean(),np.array(specificity).mean(),np.array(precision).mean(),np.array(F1).mean()])

algo={
#'NaiveBayes':GaussianNB()
#'SVM':SVC(kernel='poly',random_state=1),
#'Random Forest':RandomForestClassifier(random_state=1),
#'Logistic Regression':LogisticRegression(max_iter=1000,random_state=1),
#'AdaBoost':AdaBoostClassifier(random_state=1)
}


def k_fold_results(x,y,algo=algo):
    df=pd.DataFrame({
        #'NaiveBayes':[0,0,0,0,0],
        'SVM':[0,0,0,0,0],
        #'Random Forest':[0,0,0,0,0],
        'Logistic Regression':[0,0,0,0,0]
        #'AdaBoost':[0,0,0,0,0]
        },index=['accuracy','recall','specificity','precision','F1'])

        
    x_fold =x.to_numpy()
    y_fold=y.to_numpy()

    for item in algo.items():
        df[item[0]]=k_fold(x_fold,y_fold,item[1])
    
    return df
