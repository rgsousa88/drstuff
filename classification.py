#!/usr/bin/python3
# coding: utf-8

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from time import time
import pickle
import gridfs
import pymongo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_model_from_db(data_base,collection_name,filename):
    """
    Carrega o modelo de classficação armazenado no MongoDB  utilizando o GridFS
    A URI está configurada na variável de ambiente MONGODB_URI
    O usuário e senha (Mongo) devem ser configurados em varíavel de ambiente MONGODB_USER e MONGODB_PASSWORD
    A Collection é configurada na variável MONGODB_ML_MODEL

    Parâmetros
    data_base: a instância do banco de dados do pymongo
    collection_name: o nome da colecao onde esta armazenado o objeto a ser carregado
    filename: nome do objeto no GridFS

    Retorno
    obj: modelo de classificação treinado
    """
    fs = gridfs.GridFS(data_base,collection_name)
    obj_file = fs.find_one({"filename": filename},
                            sort=[("_id", pymongo.DESCENDING)])
    obj = obj_file.read() if obj_file else None
    obj = pickle.loads(obj)
    return obj


def load_model_from_blob(id):
    # Carrega o modelo de classficação armazendo no Azure Blob
    # Carrega um modelo de classficação criado.
    # Preciso ver quais são os parâmetros de conexão do Azure e também iremos configurar em variáveis de ambiente
    # Parâmetros
    # id Identificador do arquivo ou nome do arquivo
    # Retorno  Modelo de classificação treinado
    pass


def save_model_in_db(data_base, collection_name, filename, model, pickle_obj=True):

    """
    Armazena o objeto 'obj' no GridFS 'fs' com o nome 'filename'.
    Caso haja um arquivo com o mesmo nome, deleta a entrada anterior.
    Por padrão o objeto é serializado (pickle_obj=True)

    Salva o modelo de classficação criado No MongoDB utilizando o GridFS
    A URI está configurada na variável de ambiente MONGODB_URI
    O usuário e senha (Mongo) devem ser configurados em varíavel de ambiente MONGODB_USER e MONGODB_PASSWORD
    A Collection é configurada na variável MONGODB_ML_MODEL

    Parâmetros
    data_base: a instância do banco de dados do pymongo
    collection_name: o nome da colecao onde sera armazenado o objeto
    filename: nome do objeto no GridFS
    model: modelo a ser salvo

    """

    fs = gridfs.GridFS(data_base,collection_name)
    file = pickle.dumps(model) if pickle_obj else model
    model_file = fs.find_one({"filename": filename},
                             sort=[("_id", pymongo.DESCENDING)])
    if model_file:
        fs.delete(model_file._id)
    fs.put(file,filename=filename)


def save_model_in_blob(id, model):
    # Salva o modelo de classficação criado no AZure
    # Salva o modelo de classficação criado.
    # Preciso ver quais são os parâmetros de conexão do Azure e também iremos configurar em variáveis de ambiente
    # id Identificador do arquivo ou nome do arquivo
    # model Modelo a ser salvo
    # Retorno  Modelo de classificação treinado
    pass


def find_nest_cv_best_parameters(param_list,
                                classifier,
                                dataset,
                                target_labels,
                                cross_validation_kFold = 5,
                                scoring_metric = None,
                                n_jobs = 1):
    """
    Realiza a busca de parâmetros usando o método de Nest Cross Validation.
    Retorna a métrica alcançada com os melhores parâmetros encontrados e a
    lista de parâmetros usando validação cruzada K-Fold.

    Parametros

    paramList: quais parâmetros e os valores que serão testados no grid Search.
    Exemplo:   param_grid_gb = [{'learning_rate':[ 0.1, 0.05],
                                 'n_estimators':[30, 70, 100]}]

    classifier: instância do classificador (sklearn) que será usado para testar
    os parâmetros

    dataset: dados usados na busca de parâmetros

    target_labels: labels do dataset passado

    cross_validation_kFold: int número de K-folds a ser feita a busca de parâmetros

    scoring_metric: string com métrica diferente da default utilizada por classifier

    n_jobs: inteiro que representa o número de núcleos para paralelizar o processamento (default=1)

    O Grid Search é criado com valores de 3-fold e StratifiedKfold
    por default
    """

    gridSearch = GridSearchCV(estimator = classifier,
                              param_grid = param_list,
                              scoring = scoring_metric,
                              n_jobs = n_jobs)

    gridSearch.fit(dataset,target_labels)
    # Faço a validaçao cruzada com crossValidatinKFold-fold
    nested_score = cross_val_score(gridSearch,
                                   X = dataset,
                                   y = target_labels,
                                   cv = cross_validation_kFold)

    # Retorno o resultado da classificação com a validação cruzada de
    # crossValidatinKFold-fold e também o Objeto para recuperarmos os melhores parâmetros
    # encontrados pelo Grid Search
    return nested_score.mean(), gridSearch.best_params_


def find_gridsearch_best_parameters(param_list,
                                    classifier,
                                    dataset,
                                    target_labels,
                                    grid_search_cross_validation_kFold = 3,
                                    scoring_metric = None,
                                    n_jobs = 1):

    """
    Realiza a busca de parâmetros sem utilizar Nest Cross validation (Sem K-fold)
    Retorna lista com melhores parâmetros encontrados encontrados usando apenas o Grid Serach

    Parametros

    paramList: Lista de parâmetros e valores que serão testados no grid Search.
    Exemplo:   param_grid_gb = [{'learning_rate':[ 0.1, 0.05],
                                 'n_estimators':[30, 70, 100]}]


    classifier: instância do classificador (sklearn) que será usado para testar os parâmetros

    dataset: dados usados na busca de parâmetros

    target_labels: labels do dataset passado

    grid_search_cross_validation_kFold: int número de K-folds a ser feita a busca de parâmetros

    scoring_metric: string com métrica diferente da default utilizada por classifier

    n_jobs: inteiro que representa o número de núcleos para paralelizar o processamento (default=1)

    O Grid Search já é criado com valores de 3-fold e StratifiedKfold
    por default
    """

    gridSearch = GridSearchCV(estimator = classifier,
                              param_grid = param_list,
                              scoring = scoring_metric,
                              n_jobs = n_jobs,
                              cv = grid_search_cross_validation_kFold)

    gridSearch.fit(dataset,target_labels)

    return gridSearch.best_params_



def create_model(classifier,
                 dataset,
                 target_labels,
                 param_list=None):
    """
    Cria/treina um modelo a partir do classificador e parâmetros passados

    Parametros

    paramList: Lista de parâmetros e valores utilizados no classificador.
    Exemplo:   param_list = [{"C": 10,
                             "gamma":1,
                             "kernel":"rbf"}]

    classifier: instância do classificador (sklearn) possuindo o método fit_transform

    dataset: dados usados no treino do modelo

    target_labels: labels do dataset passado

    """
    if(param_list!=None):
        for param in param_list:
            for parameter, value in param.items():
                classifier.__setattr__(parameter,value)

    t0 = time()
    classifier.fit(dataset,target_labels)
    print("Model trained in %0.3fs"%(time()-t0))

    return classifier


def apply_model(classifier,test_data, proba=False):
    """
    Aplica um modelo sobre o conjunto de teste passado

    Parametros

    classifier: instância do classificador (sklearn) possuindo o método predict

    teste_data: conjunto de teste

    """
    if(proba):
        return classifier.predict_proba(test_data)
    else:
        return classifier.predict(test_data)


def compute_metrics(true_labels,
                    predicted_labels,
                    metric="all",
                    visualize_CM=False):
    """
    Calcula e retorna a metrica definida em metric comparado predicted_labels
    (valores obtidos pela classificacao) com true_labels (ground truth).

    Parametros

    true_labels: ground truth, labels reais do dados

    predicted_labels: labels obtidos pelo método predict do classificador

    metric: string definindo a metrica a ser utilizada (default "all").
            Valores possíveis: "accuracy", "recall", "precision", "f1" e "all"

    visualize_CM: boolean para selecionar se a matrix de confusão do modelo
                  será apresentada (default=False)

    Retorno

    Dicionário contendo a métrica (key) e seu respectivo valor (value)

    """

    if(metric=='recall'):
        recall = recall_score(true_labels,predicted_labels)
        score = [{metric:recall}]
    elif(metric=='precision'):
        precision = precision_score(true_labels,predicted_labels)
        score = [{metric:precision}]
    elif(metric=='f1'):
        f1 = f1_score(true_labels,predicted_labels)
        score = [{metric:f1}]
    elif(metric=='accuracy'):
        accuracy = accuracy_score(true_labels,predicted_labels)
        score = [{metric:accuracy}]
    elif(metric=='all'):
        recall = recall_score(true_labels,predicted_labels)
        precision = precision_score(true_labels,predicted_labels)
        f1 = f1_score(true_labels,predicted_labels)
        accuracy = accuracy_score(true_labels,predicted_labels)
        score = [{"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1}]
    else:
        print("Invalid metric %s"%(metric))
        return None

    if(visualize_CM):
        cm = confusion_matrix(true_labels,predicted_labels)
        print(cm)

    return score



def apply_cross_validation(classifier,
                           dataset,
                           target_labels,
                           k_fold=3,
                           param_list=None):

    """
    Realiza cross validação e retorna as métricas encontradas

    Parametros

    k_fold: int número de K-folds a ser feita a busca de parâmetros (default = 3)

    paramList: lista de parâmetros e valores que serão testados (default = None).
    Exemplo para uma SVM:   param_list = [{'C':0.1,
                                           'gamma':10,
                                           'kernel':'rbf'}]

    classifier: instância do classificador a ser validado, necessita ter método fit_transform

    dataset: dados usados no treino do modelo

    target_labels: labels do dataset passado

    Retorno

    Valor da métrica utilizada no teste de cross validação.

    """

    if(param_list!=None):
            for param in param_list:
                for parameter, value in param.items():
                    classifier.__setattr__(parameter,value)

    results = cross_validate(estimator=classifier,
                             X=dataset,
                             y=target_labels,
                             cv=k_fold)

    return results['test_score']
