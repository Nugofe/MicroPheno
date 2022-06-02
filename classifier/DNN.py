__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "2.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"


import sys
sys.path.append('../')

import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras import utils as np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from utility.file_utility import FileUtility
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from gensim.models.wrappers import FastText
import itertools
import os
import matplotlib.pyplot as plt
import matplotlib


class DNNMutliclass16S(object):
    '''
    Deep MLP Neural Network
    '''

    # NUESTRA MLP
    # - MLP: tipo de DNN con varias hidden layers
    # - RELU: función de activación genérica
    # - softmax: función de activación de la última capa
    # - early stopping y droppout para evitar overfitting
    # (weights less than one are dropouts in model_arch and the rest the sizes of the hidden layers)
    
    def __init__(self, X,Y, model_strct='mlp', model_arch=[500]): # model_arch=[1024,0.2,512,0.2,256,0.1,128,8]   num > 0 : sieze of hidden layer,   num < 0 : dropout
        self.X=X
        self.Y=Y

        encoder = LabelEncoder()
        encoder.fit(Y)
        self.C=len(set(self.Y))  # número de tipos de clasificación (2 -> CD, Not-CD)
        
        self.encoded_Y = encoder.transform(Y) # transform non-numerical labels (as long as they are hashable) to numerical labels
        self.onehot_y = np_utils.to_categorical(self.encoded_Y) # return a matrix of binary values (either ‘1’ or ‘0’). (rows = length of the input vector, columns = number of classes).
        
        self.model_strct=model_strct # mlp
        self.model_arch=model_arch
    
    def get_MLP_model(self):
        '''
        Create the model
        :return:
        '''
        # creating the model
        # objeto en el que construimos el modelo: se irán añadiendo las capas una a una (según las vayamos creando en el for)
        # [adds a new layer on top of the layer stack]
        model = Sequential()
        for layer_idx, h_layer_size in enumerate(self.model_arch): # layer_idx = num of current iteration, h_layer_size = value of the item at the current iteration
            if layer_idx==0:
                # Dense: output num (valores en model_arch), input num (numero muestras = 1359) -> como es la primera, solo esta necesita definir inputs (el resto se conecta todo con todo)
                model.add(Dense(h_layer_size, input_dim=self.X.shape[1], activation='relu'))
            else:
                if h_layer_size < 1: # las que tienen un valor menor que 1, son dropouts
                    # Dropout: rate or fraction of the input units to drop (value between 0 and 1)
                    model.add(Dropout(h_layer_size))
                else:
                    model.add(Dense(h_layer_size, activation='relu'))
        model.add(Dense(self.C, activation='softmax'))
        # Compile model 
        # función de coste = cross entropy, optimizador de backpropagation = adam, juzgar la calidad del modelo = accuracy
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #print('INICIO MODEL SUMARY!!!!!\n' + str(model.summary()) +  '\nFIN MODEL SUMMARY!!!!!|n')
        return model

    # ESTA NO
    def get_pretrained_model(self, file_name, trainable):
        pretrained_weights=FileUtility.load_obj(file_name)
        
        h_sizes=[float(x) for x in file_name.split('/')[-1].split('_')[3].split('-')]
        model = Sequential()
        for layer_idx, h_layer_size in enumerate(h_sizes):
            if layer_idx==0:
                model.add(Dense(int(h_layer_size), input_dim=self.X.shape[1], weights=pretrained_weights[0],  activation='relu', trainable=trainable))
            else:
                if h_layer_size < 1:
                    model.add(Dropout(h_layer_size, weights=pretrained_weights[layer_idx], trainable=trainable))
                else:
                    model.add(Dense(int(h_layer_size), weights=pretrained_weights[layer_idx], activation='relu', trainable=trainable))
        if self.model_arch:
            for layer_idx, h_layer_size in enumerate(self.model_arch):
                if h_layer_size < 1:
                    model.add(Dropout(h_layer_size))
                else:
                    model.add(Dense(h_layer_size, activation='relu'))
        model.add(Dense(self.C, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        
    
    # volver a crear el modelo, pero ahora entrenado en con todos los datos para guardar los pesos y las predicciones
    def make_weights(self, file_path, epochs, batch_size, f1mac):
        if self.model_strct=='mlp':
            model=self.get_MLP_model()
        
        model.fit(self.X, self.onehot_y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

        weights=[]
        for x in model.layers:
            weights.append(x.get_weights())

        self.weights_file_name=file_path + '/weights_' + '_'.join(['layers', self.model_strct,'-'.join([str(x) for x in self.model_arch]), str(np.round(f1mac,2))])
        FileUtility.save_obj(self.weights_file_name,  weights)
        self.weights_file_name=self.weights_file_name + '.pickle'


    def tune_and_eval(self, file_path, gpu_dev='2', n_fold=10, epochs=50, batch_size=100, pretrained_model=False, trainable=False):
        '''
        :param file_path:
        :param gpu_dev:
        :param n_fold: k groups
        :param epochs: The number of times that the learning algorithm will work through the entire training set
        :param batch_size: The number of samples of the training set to work through before updating the internal model parameters
        :param model_strct: Model structure (MLP in our case)
        :param pretrained_model:
        :param trainable:
        :return:
        '''
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev # CPUs ???
        
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True) # CV con k = 10

        p_micro=[] # cada elemento es la métrica calculada en un fold
        p_macro=[]
        r_micro=[]
        r_macro=[]
        f1_micro=[]
        f1_macro=[]
        accuracy=[]
        roc_auc=[]

        for train_index, valid_index in skf.split(self.X, self.Y): # Generate indices to split data into training and validation set
            print ('\n Evaluation on a new fold is now get started ..')
            X_train=self.X[train_index,:]
            y_train=self.onehot_y[train_index,:]
            y_class_train=self.encoded_Y[train_index] # clasificación real (0, 1)
            X_valid=self.X[valid_index,:]
            y_valid=self.onehot_y[valid_index,:]
            y_class_valid=self.encoded_Y[valid_index]
            
            if pretrained_model:
                model=self.get_pretrained_model(self.model_strct, trainable)
            else: # ESTA SI
                if self.model_strct=='mlp':
                    model=self.get_MLP_model() # modelo con sus capas, pesos, funciones de activación...
            
            # fitting: entrenar el modelo
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_valid, y_valid), verbose=0)
            pred = model.predict_classes(X_valid) # Generate class predictions for the input samples batch by batch for the validation set (como el batch ya se estableció antes, aquí ya no se indica)
            
            # score-calculations
            f1_micro.append(f1_score(y_class_valid,pred, average='micro'))
            f1_macro.append(f1_score(y_class_valid,pred, average='macro'))
            p_micro.append(precision_score(y_class_valid,pred, average='micro'))
            p_macro.append(precision_score(y_class_valid,pred, average='macro'))
            r_micro.append(recall_score(y_class_valid,pred, average='micro'))
            r_macro.append(recall_score(y_class_valid,pred, average='macro'))
            accuracy.append(accuracy_score(y_class_valid,pred))
            roc_auc.append(roc_auc_score(y_class_valid,pred))

        # guardar las métricas obtenidas
        # mean values
        f1mac=np.mean(f1_macro)
        f1mic=np.mean(f1_micro)
        prmac=np.mean(p_macro)
        prmic=np.mean(p_micro)
        remac=np.mean(r_macro)
        remic=np.mean(r_micro)
        maccur=np.mean(accuracy)
        mrocauc=np.mean(roc_auc)
        # std values - desviación típica
        sf1mac=np.std(f1_macro)
        sf1mic=np.std(f1_micro)
        sprmac=np.std(p_macro)
        sprmic=np.std(p_micro)
        sremac=np.std(r_macro)
        sremic=np.std(r_micro)
        saccur=np.std(accuracy)
        srocauc=np.std(roc_auc)
        # table
        latex_line=' & '.join([str(np.round(x,2))+' $\\pm$ '+str(np.round(y,2)) for x,y in [ [prmic, sprmic], [remic, sremic], [f1mic, sf1mic], [prmac, sprmac], [remac, sremac], [f1mac, sf1mac], [maccur, saccur], [mrocauc, srocauc] ]])      
        
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        '''Saving the metrics:   results_mlp_1024-0.2-512-0.2-256-0.1-128-8_0.65'''
        if pretrained_model:
            self.model_strct='pretrained'

                            # file name                                                                                                                   # results to save
        FileUtility.save_obj(file_path + '/results_' + '_'.join([self.model_strct,'-'.join([str(x) for x in self.model_arch]), str(np.round(f1mac,2))]),  [latex_line, p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, accuracy, roc_auc, (loss_values, val_loss_values, epochs)]) 


        # guardar los datos más importantes en un formato más leíble
        attributes=['model_arch: ' + str(self.model_arch),
                    'mean_f1_macro: ' + str(f1mac),        'mean_f1_micro: ' + str(f1mic), 
                    'mean_precision_macro: ' + str(prmac), 'mean_precision_micro: ' + str(prmic),
                    'mean_recall_macro: ' + str(remac),    'mean_recall_micro: ' + str(remic),
                    'mean_accuracy: ' + str(maccur),       'mean_roc_auc: ' + str(mrocauc),
                    'std_f1_macro: ' + str(sf1mac),        'std_f1_micro: ' + str(sf1mic), 
                    'std_precision_macro: ' + str(sprmac), 'std_precision_micro: ' + str(sprmic),
                    'std_recall_macro: ' + str(sremac),    'std_recall_micro: ' + str(sremic),
                    'std_accuracy: ' + str(saccur),        'std_roc_auc: ' + str(srocauc)]

        FileUtility.save_text_array(file_path+'/best_metrics.txt', attributes)
        
        # calcular los pesos de la red al entrenarla con todos los datos
        self.make_weights(file_path, epochs, batch_size, f1mac)


    # crear modelo con la función de activación final
    def make_activation_function(self, X=None, file_name=None, last_layer=None): # filename = weights_layers_mlp_1024-0.2-512-0.2-256-0.1-128-8_0.65
        if file_name is None:
            file_name=self.weights_file_name
        if X is None:
            X=self.X
        
        pretrained_weights=FileUtility.load_obj(file_name)
        if last_layer:
            h_sizes=[float(x) for x in file_name.split('/')[-1].split('_')[3].split('-')]+[last_layer]
        else:
            h_sizes=[float(x) for x in file_name.split('/')[-1].split('_')[3].split('-')] # ESTE SI (h_sizes = tamaños de las capas que se calcularon en cross_validation; basicamente el model_arch)

        model = Sequential()
        for layer_idx, h_layer_size in enumerate(h_sizes):
            if layer_idx==0:
                model.add(Dense(int(h_layer_size), input_dim=X.shape[1], weights=pretrained_weights[0],  activation='relu'))
            else:
                if h_layer_size < 1:
                    model.add(Dropout(h_layer_size, weights=pretrained_weights[layer_idx]))
                else:
                    if layer_idx == len(h_sizes)-1 and last_layer: # en nuestro estudio, ésta no la va a hacer nunca 
                        model.add(Dense(int(h_layer_size), weights=pretrained_weights[layer_idx], activation='softmax'))
                    else:
                        model.add(Dense(int(h_layer_size), weights=pretrained_weights[layer_idx], activation='relu'))
        activations = model.predict(X) # Generates output predictions for the input samples.

        '''Saving the activation layer results:   activationlayer_layers_mlp_1024-0.2-512-0.2-256-0.1-128-8_0.65'''
        np.savetxt(file_name.replace(file_name.split('/')[-1].split('_')[0],'activationlayer'),  activations)
        
        return activations
    
    
    @staticmethod
    def result_visualization(filename):
        [latex_line, p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, accuracy, roc_auc, (loss_values, val_loss_values, epochs)]=FileUtility.load_obj(filename)
        print(latex_line)

    @staticmethod
    def load_history(filename, fileout):
        '''
        Plot the history
        :param filename:
        :param fileout:
        :return:
        '''
        [latex_line, p_micro, r_micro, f1_micro, p_macro, r_macro, f1_macro, history]=FileUtility.load_obj(filename)
        (loss_values, val_loss_values, epochs)=history
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams['mathtext.fontset'] = 'custom'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        matplotlib.rcParams["axes.edgecolor"] = "black"
        matplotlib.rcParams["axes.linewidth"] = 0.6
        plt.rc('text', usetex=True)
        plt.plot(epochs, loss_values, 'ro', label='Loss for train set')
        plt.plot(epochs, val_loss_values, 'b+', label='Loss for test set')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc=1, prop={'size': 8},ncol=1, edgecolor='black', facecolor='white', frameon=True)
        plt.title('Loss with respect to the number of epochs for train and test sets')
        plt.savefig(fileout+'.pdf')
        plt.show()


# esto se hace en el notebook en las 2 primeras celdas
def crohns_disease():
    '''
    '''
    #[1024,0.2,256,0.1,256,0.1,128,0.1,64]
    X=FileUtility.load_sparse_csr('../../datasets/processed_data/crohn/sample-size/6-mers_rate_complete1359_seq_5000.npz').toarray()
    Y=FileUtility.load_list('../../datasets/processed_data/crohn/data_config/labels_disease_complete1359.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[512,0.2,256,0.2,128,0.1,64,16])
    DNN.cross_validation('../../datasets/results/crohn/classifier/nn', gpu_dev='2', n_fold=3, epochs=25, batch_size=10, model_strct='mlp')

# ESTAS NO
def bodysite():
    X=FileUtility.load_sparse_csr('../MicroPheno_datasets/body-sites/k-mer_representations_labels/6-mers_rate_5000.npz').toarray()
    Y=FileUtility.load_list('../MicroPheno_datasets/body-sites/k-mer_representations_labels/labels_phen.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[512,0.2,256,0.2,128,0.1,64])
    DNN.cross_validation('../MicroPheno_datasets/body-sites/nn', gpu_dev='2', n_fold=3, epochs=300, batch_size=10, model_strct='mlp')

def eco_classification():
    '''
    '''
    #[1024,0.2,256,0.1,256,0.1,128,0.1,64]
    for d1 in [500]:
        X=FileUtility.load_sparse_csr('../../datasets/processed_data/eco/K/6-mer_eco_restrictedmer.npz').toarray()
        Y=FileUtility.load_list('../../datasets/processed_data/eco/K/eco_label_restrictedkmer.txt')
        DNN=DNNMutliclass16S(X,Y,model_arch=[2048,0.2,128])
        DNN.cross_validation('../../datasets/results/eco/classifier/nn', gpu_dev='2', n_fold=5, epochs=30, batch_size=100, model_strct='mlp')

def eco_10000_classification():
    '''
    '''
    #[1024,0.2,256,0.1,256,0.1,128,0.1,64]
    X=FileUtility.load_sparse_csr('../../datasets/processed_data/eco_10000/K/6-mer_eco_restrictedmer.npz').toarray()
    Y=FileUtility.load_list('../../datasets/processed_data/eco_10000/K/eco_label_restrictedkmer.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[1024,0.2,1024,0.2,512,0.2,512,0.1,128,0.1,64])
    DNN.cross_validation('../../datasets/results/eco_10000/classifiers/nn', gpu_dev='1', n_fold=5, epochs=10, batch_size=1000, model_strct='mlp')

def eco_all_classification():
    '''
    '''
    #[1024,0.2,256,0.1,256,0.1,128,0.1,64]
    X=FileUtility.load_sparse_csr('../../datasets/processed_data/eco_all_classes/6-mer_eco_restrictedmer_all.npz').toarray()
    Y=FileUtility.load_list('../../datasets/processed_data/eco_all_classes/eco_label_restrictedkmer_all.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[1024,0.2,512,0.2,512,0.1,256])
    DNN.cross_validation('../../datasets/results/eco_all/nn', gpu_dev='1', n_fold=10, epochs=20, batch_size=10, model_strct='mlp')
    
def eco_all_classification_transfer_learning():
    '''
    '''
    #[1024,0.2,256,0.1,256,0.1,128,0.1,64]
    X=FileUtility.load_sparse_csr('../../datasets/processed_data/eco_all_classes/6-mer_eco_restrictedmer_all.npz').toarray()
    Y=FileUtility.load_list('../../datasets/processed_data/eco_all_classes/eco_label_restrictedkmer_all.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[512,0.1,256, 0.1,128])
    DNN.cross_validation('../../datasets/results/eco_all/nn', gpu_dev='6', pretrained_model=True,trainable=False, n_fold=5, epochs=10, batch_size=10, model_strct='../../datasets/results/eco_10000/classifiers/nn_layers_mlp_1024-0.2-512-0.2-512_0.88.pickle')

def org_classification():
    '''
    '''
    X=FileUtility.load_sparse_csr('../../datasets/processed_data/org/K/6-mer_org_restrictedkmer.npz').toarray()
    Y=FileUtility.load_list('../../datasets/processed_data/org/K/org_label_restrictedkmer.txt')
    DNN=DNNMutliclass16S(X,Y,model_arch=[1024,0.2,256,0.1,256,0.1,128,0.1,64])
    DNN.cross_validation('../../datasets/results/org/classifier/nn', gpu_dev='2', n_fold=10, epochs=30, batch_size=100, model_strct='mlp')

if __name__=='__main__':
    bodysite()

