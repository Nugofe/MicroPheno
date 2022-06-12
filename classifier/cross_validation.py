__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "2.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"

import sys
sys.path.append('../')
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from utility.file_utility import FileUtility
from sklearn.metrics import f1_score,confusion_matrix
import pandas as pd
import time


class CrossValidator(object):
    '''
     The Abstract Cross-Validator
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.scoring =  {
                            'precision_micro': 'precision_micro', 
                            'precision_macro': 'precision_macro', 
                            'recall_macro': 'recall_macro',
                            'recall_micro': 'recall_micro', 
                            'f1_macro': 'f1_macro', 
                            'f1_micro': 'f1_micro',
                            'accuracy': 'accuracy',
                            'roc_auc': 'roc_auc'
                        }


class KFoldCrossVal(CrossValidator):
    '''
        K-fold cross-validation tuning and evaluation
    '''
    def __init__(self, X, Y, labels_num, labels, folds=10, random_state=1):
        '''
        :param X:
        :param Y:
        :param folds:
        :param random_state:
        '''
        CrossValidator.__init__(self, X, Y)

        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        # - n_splits: Number of folds. Must be at least 2.  (k=10)
        # - shuffle: Whether to shuffle each class’s samples before splitting into batches. 
        #   Note that the samples within each split will not be shuffled.
        # - random_state: When shuffle is True, random_state affects the ordering of the indices, 
        #   which controls the randomness of each fold for each class. Otherwise, leave random_state as None.
        self.cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        self.X = X
        self.Y = Y
        self.labels_num = labels_num
        self.labels = labels

    # ejecutamos el CV, entrenamos el modelo con los parámetros óptimos y guardamos los resultados
    def tune_and_evaluate(self, estimator, parameters, score='macro_f1', n_jobs=-1, file_path='results'):
        '''
        :param estimator:
        :param parameters:p
        :param score:
        :param n_jobs:
        :param file_name: directory/tuning/classifier/features/
        :return:
        '''
        # definir greed_search
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # - estimator: o modelo ou clasificador que empreguemos.
        # - param_grid: os parámetros que lle pasasemos.
        # - cv: cross-validation splitting strategy (StratifiedKFold no noso caso).
        # - scoring: Strategy to evaluate the performance of the cross-validated model on the test set (f1_macro no noso caso).
        # - refit: Refit (reaxustar) an estimator using the best found parameters on the whole dataset.
        # - error_score: Value to assign to the score if an error occurs in estimator fitting.
        # - n_jobs: Number of jobs to run in parallel.
        start_time = time.time()
        
        self.greed_search = GridSearchCV(estimator=estimator, param_grid=parameters, cv=self.cv, scoring=self.scoring,
                                         refit=score, error_score=0, n_jobs=n_jobs)

        # ejecutar greed_search (fitting)
        self.greed_search.fit(X=self.X, y=self.Y)

        # entrenar el modelo con los mejores parámetros
        y_predicted = cross_val_predict(self.greed_search.best_estimator_, self.X, self.Y)

        end_time = time.time() - start_time

        conf=confusion_matrix(self.Y, y_predicted, labels=self.labels_num)

        # guadar resultados en un .pickle
        # - label_set: CD, Not-CD
        # - conf: matriz de confusión
        # - best_score_: Mean cross-validated score of the best_estimator  -                            ME INTERESA
        # - best_estimator: RF o SVM que obtuvo los mejores resultados (con sus atibutos y parámetros)
        # - cv_results_: métricas resultado  -                                                          ME INTERESA
        # - best_params_: parámetros que produjeron los mejores resultados  -                           ME INTERESA
        # - y_predicted: resultados de la clasificación, que no los reales
        FileUtility.save_obj(file_path+'/all_results', [self.labels, conf, self.greed_search.best_score_, 
                                        self.greed_search.best_estimator_, self.greed_search.cv_results_, 
                                        self.greed_search.best_params_, y_predicted])


        # guardar los resultados más importantes en otro tipo de ficheros para que verlos más facilmente
        # explicación atributos: https://stackoverflow.com/questions/54608088/what-is-gridsearch-cv-results-could-any-explain-all-the-things-in-that-i-e-me
        
        # cv_results_: Each row of this dataframe gives the gridsearch metrics for one combination of the parameters
        # RF: "n_estimators": [100, 200, 500, 1000]     SVM: 'C': [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.2, 0.5, 0.01, 0.02, 0.05, 0.001]
        df = pd.DataFrame(self.greed_search.cv_results_)
        df.to_csv(file_path + '/all_metrics.csv') # cv_results_ completo

        
        df = df.loc[df['params'] == self.greed_search.best_params_] # coger la fila de resultados del mejor estimador
        df = df.iloc[: , 1:] # eliminar la primera columna (unnamed)
        df = df[df.columns.drop(list(df.filter(regex='time|param|params|rank|split')))] # eliminar las columnas que no nos interesan
        attributes=[
                    'best_params_: ' + str(self.greed_search.best_params_),
                    'cross_val_time: ' + str(end_time)
                    ]
        for (column, value) in zip(df.columns.tolist(), df.values[0].tolist()):
            attributes.append(column + ': ' + str(value))

        FileUtility.save_text_array(file_path+'/best_metrics.txt', attributes)
