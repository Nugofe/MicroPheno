__author__ = "Ehsaneddin Asgari"
__license__ = "Apache 2"
__version__ = "2.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu or ehsaneddin.asgari@helmholtz-hzi.de"
__project__ = "LLP - MicroPheno"
__website__ = "https://llp.berkeley.edu/micropheno/"



import sys
sys.path.append('../')
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import numpy as np
from multiprocessing import Pool
import tqdm
import random
from scipy import sparse
from utility.file_utility import FileUtility
from Bio import SeqIO
import timeit
import pandas as pd


class Metagenomic16SRepresentation:
    '''
        Make k-mer from directory of fasta files
    '''

    def __init__(self, fasta_files, indexing, sampling_number=3000, num_p=20):
        '''
        :param fasta_files: list of fasta files
        :param indexing: the index
        :param sampling_number: value N
        :param num_p: number of cores/processors
        '''
        self.fasta_files=fasta_files
        self.num_p=num_p
        self.sampling_number=sampling_number
        self.indexing=indexing

    # NO SE USA
    def get_corpus(self, file_name_sample):
        '''
        :param file_name_sample:
        :return:
        '''
        file_name=file_name_sample[0]
        sample_size=file_name_sample[1]
        corpus=[]
        if file_name[-1]=='q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        return file_name, random.sample(corpus, min(sample_size,len(corpus))) 
    

    def generate_kmers_all(self, k, save=False):
        '''
        :param k:
        :param save:
        :return:
        '''

        # generar todas las posibles cadenas compuestas por a,c,g,t de tamaño k
        # Cadenas generadas = 4^k; donde 4 es el número de nucleótidos y k el tamaño de la cadena (k-mer)
        #   Ejemplo: k=2 => 16 cadenas: ['aa', 'at', 'ac', ...]     k=3 => 64 cadenas
        self.k=k
        self.vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
        self.vocab.sort()
        
        # convertir una colección de documentos en una matriz TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        # - use_idf=False : no se aplica la frecuencia inversa y se le da mayor peso a las letras que se repiten más
        # - ngram_range=(k, k) : grupos de k
        self.vectorizer = TfidfVectorizer(use_idf=False, vocabulary=self.vocab, analyzer='char', ngram_range=(k, k),
                                          norm=None, stop_words=[], lowercase=True, binary=False)

        # matriz de ceros (nº filas = nº archivos fasta, nº columnas = nº combinaciones k-mers)
        data = np.zeros((len(self.fasta_files), len(self.vocab))).astype(np.float64)

        # extracción de las k-mer distributions
        # - pool.imap_unordered : ejecución en num_p procesadores/cores
        # - tqdm.tqdm : mostrar barra de progreso
        t_steps=[]
        s_steps=[]
        pool = Pool(processes=self.num_p)
        for ky, (v,t,s) in tqdm.tqdm(pool.imap_unordered(self.get_kmer_distribution, self.fasta_files, chunksize=1),
                               total=len(self.fasta_files)):
            data[self.indexing[ky], :] = v
            t_steps.append(t)
            s_steps.append(s)

        # normalizar las frecuencias: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
        # - axis : axis used to normalize the data along.
        #   If axis=1, independently normalize each sample; otherwise, normalize each feature.
        # - norm=l1 : usar técnica de normalización l1 (the sum of the absolute values of the vector)
        #   L1 is useful for feature selection, while L2 is useful when you have collinear/codependent features.
        #   https://explained.ai/regularization/L1vsL2.html
        data = normalize(data, axis=1, norm='l1')
        data = sparse.csr_matrix(data)

        # en la carpeta datasets, guardar los resultados (1 por cada combinación N k).
        # - .npz =>  resultados / k-mer distributions
        # - _meta => rutas a archivos fastaq                       SON TODOS IGUALES, PERO PARA QUÉ ME VALE ESO?
        # - _log =>  media y desviación típica (mean and std)
        if save:
            FileUtility.save_sparse_csr(save, data)
            #FileUtility.save_list(save+'_meta',self.fasta_files)
            #FileUtility.save_list(save+'_log',[': '.join(['mean_size', str(np.mean(s_steps))]), ': '.join(['std_size', str(np.std(s_steps))])])
        

        # guardamos en un .csv la k-mer distribution para verla mejor
        df = pd.DataFrame.sparse.from_spmatrix(data)
        df.to_csv(save + '.csv')

        return data

    def get_kmer_distribution(self, file_name):
        '''
        :param file_name:
        :return:
        '''
        start = timeit.timeit()
        corpus=[]
        if file_name[-1]=='q':
            for cur_record in SeqIO.parse(file_name, "fastq"):
                corpus.append(str(cur_record.seq).lower())
        else:
            for cur_record in SeqIO.parse(file_name, "fasta"):
                corpus.append(str(cur_record.seq).lower())
        tot_size=len(corpus)
        if self.sampling_number==-1:
            random.shuffle(corpus)
        else:
            corpus = random.sample(corpus, min(self.sampling_number,len(corpus)))
        end = timeit.timeit()
        return file_name, (np.sum(self.vectorizer.fit_transform(corpus).toarray(), axis=0),end - start,tot_size)  # np sum calcula la abundancia


class FastaRepresentations(object):
    '''
        Make k-mer from single fasta file
        where the headers contain info about the label
    '''
    def __init__(self, fasta_address, label_modifying_func=str):
        '''
        :param fasta_address:
        :param label_modifying_func: extract label from the header
        '''
        self.labels=[]
        self.corpus=[]
        for cur_record in SeqIO.parse(fasta_address, 'fasta'):
            self.corpus.append(str(cur_record.seq).lower())
            self.labels.append(str(cur_record.id).lower())
        self.labels=[label_modifying_func(l) for l in self.labels]

    def get_samples(self, envs, N):
        '''
        :param envs: list of labels
        :param N: sample size
        :return: extract stratified with size N corpus and label list
        '''
        labels=[]
        corpus=[]
        for env in envs:
            selected=[idx for idx,v in enumerate(self.labels) if env==v]
            if N==-1:
                N=len(selected)
            idxs=random.sample(selected, N)
            corpus=corpus+[self.corpus[idx] for idx in idxs]
            labels=labels+[self.labels[idx] for idx in idxs]
        return corpus, labels

    def get_vector_rep(self, corpus, k, restricted=True):
        '''
        :param corpus:
        :param k: k-mer size
        :param restricted: restricted to known values
        :return:
        '''
        if restricted:
            vocab = [''.join(xs) for xs in itertools.product('atcg', repeat=k)]
            tf_vec = TfidfVectorizer(use_idf=True, vocabulary=vocab, analyzer='char', ngram_range=(k, k),
                                                  norm='l1', stop_words=[], lowercase=True, binary=False)
        else:
            tf_vec = TfidfVectorizer(use_idf=True, analyzer='char', ngram_range=(k, k),
                                                  norm='l1', stop_words=[], lowercase=True, binary=False)
        return tf_vec.fit_transform(corpus)

if __name__=='__main__':
    FR=FastaRepresentations('sample.fasta')
    MR=Metagenomic16SRepresentation('16ssamples/')


# PROBAR CON 2 FASTAQ, N = 100 y varios k