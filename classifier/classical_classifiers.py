__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "2.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import sys

sys.path.append('../')
from utility.file_utility import FileUtility
from classifier.cross_validation import KFoldCrossVal
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

# Random Forest
class RFClassifier:
    def __init__(self, X, Y, labels=None):

        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # - bootstrap: Whether bootstrap samples are used when building trees. 
        #   If False, the whole dataset is used to build each tree.
        # - criterion: The function to measure the quality of a split.
        # - min_samples_split: The minimum number of samples required to split an internal node.
        # - max_features: The number of features to consider when looking for the best split.
        #   If “auto”, then `max_features=sqrt(n_features)`
        # - min_samples_leaf: The minimum number of samples required to be at a leaf node.
        # - n_estimators: The number of trees in the forest.
        self.model = RandomForestClassifier(bootstrap=True, criterion='gini',
                                            min_samples_split=2, max_features='auto', min_samples_leaf=1,
                                            n_estimators=1000) # esta configuración non serve para nada, éralles mellor inicializalo baleiro
        
        if labels:                   # labels en formato letra, si se las he pasado
            self.labels=labels       # le paso yo el orden qque quiero
        else:
            self.labels=list(set(Y)) # me da igual cual sea la clase positiva y cual la negativa

        self.X = X
        self.Y = FileUtility.encode_labels(Y, self.labels)

        self.labels_num=list(set(self.Y))  # labels en formato número:  1 = CD, 0 = Not-CD
        self.labels_num.sort()             # para que se ponga primero el 0
        self.C=len(self.labels_num)        # número de tipos de clasificación (2 -> CD, Not-CD)
        
        print('labels_num  !!!!!!!!!!!' + str(self.labels_num))
        print('labels      !!!!!!!!!!!' + str(self.labels))


    def tune_and_eval(self, results_file_path, params=None, n_fold=10, n_jobs=1): #n_jobs=15
        # búsqueda de parámetros óptimos
        if params is None:
            params = [{"n_estimators": [100, 200, 500, 1000],
                       "criterion": ["entropy"],  # "gini",
                       'max_features': ['sqrt'],  # 'auto',
                       'min_samples_split': [5],  # 2,5,10
                       'min_samples_leaf': [1]}]

        self.CV = KFoldCrossVal(self.X, self.Y, self.labels_num, self.labels, folds=n_fold) # CV = K-fold Cross Validation (k=10)
        self.CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_path=results_file_path, n_jobs=n_jobs)


# Support Vector Machine
class SVM:
    def __init__(self, X, Y, labels=None, clf_model='LSVM'):
        # escoger el tipo de SVM
        if clf_model == 'LSVM': # ESTA SI
            # [https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
            # - C: Regularization parameter.
            # - multi_class: Multi-class strategy if `y` contains more than two classes.
            #   `"ovr"` trains n_classes one-vs-rest classifiers.
            self.model = LinearSVC(C=1.0, multi_class='ovr')
            self.type = 'linear'
        else:
            self.model = SVC(C=1.0, kernel='rbf')
            self.type = 'rbf' # Radial Basis Function kernel
        
        if labels:                   # labels en formato letra, si se las he pasado
            self.labels=labels       # le paso yo el orden qque quiero
        else:
            self.labels=list(set(Y)) # me da igual cual sea la clase positiva y cual la negativa

        self.X = X
        self.Y = FileUtility.encode_labels(Y, self.labels)

        self.labels_num=list(set(self.Y))  # labels en formato número:  1 = CD, 0 = Not-CD
        self.labels_num.sort()             # para que se ponga primero el 0
        self.C=len(self.labels_num)        # número de tipos de clasificación (2 -> CD, Not-CD)
        
        print('labels_num  !!!!!!!!!!!' + str(self.labels_num))
        print('labels      !!!!!!!!!!!' + str(self.labels))


    # ejecutar el entrenamiento
    def tune_and_eval(self, results_file_path,
                      params=[{'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 
                                     1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001]}], n_fold=10, n_jobs=1):  #n_jobs=10
        
        # hacer Cross Validation
        CV = KFoldCrossVal(self.X, self.Y, self.labels_num, self.labels, folds=n_fold)
        CV.tune_and_evaluate(self.model, parameters=params, score='f1_macro', file_path=results_file_path, n_jobs=n_jobs)
