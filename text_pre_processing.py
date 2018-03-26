#!/usr/bin/python3
# coding: utf-8

import re
from nltk import tokenize
from nltk.stem.snowball import PortugueseStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def remove_stop_words(data, stop_words):
    """
    Remove as stop words da string de entrada

    Parâmetros
    data: string com texto de entrada
    stop_words: lista de stop_words

    Retorno
    String com texto sem stop words
    """

    words = data.split()
    stop_words_free = [word for word in words if word.lower() not in stop_words]
    stop_words_free = " ".join(stop_words_free)
    return stop_words_free

def remove_puntuaction(data):
    """
    Remove pontuação da string de entrada

    Parâmetros
    Data: string com texto de entrada

    Retorno
    String com texto sem pontuação
    """

    data = re.sub(u'\xa0',' ',data)
    noise_free = re.sub(r'[^a-zA-Z0-9à-úÀ-Ú\s]','',data)

    return noise_free

def remove_by_regex(data, list_regex):
    """
    Remove pontuação da string de entrada

    Parâmetros
    Data: string com texto de entrada

    Retorno
    String com texto aplicado a regex
    """

    for regex in list_regex:
        data = re.sub(regex, '', data)

    return data

def remove_emails_and_URLs(data):
    """
    Remove URL e emails

    Parâmetros
    data: string com texto de entrada

    Retorno
    String com texto sem emails e URLs
    """

    url_free = re.sub(r'(?i)(www[.a-z\/:,_#?!&=]+|http[.a-z\/:\d,_#?!\-&=]+)','',data)
    email_url_free = re.sub(r'(?i)([a-z0-9.,_\-]+@[a-z0-9.\-_]+)','',url_free)

    return email_url_free


def get_list_of_sentences(data,language='portuguese',regex=None):
    """
    Quebra o texto de entrada em uma lista das sentenças/frases

    Parâmetros
    data: string com texto de entrada
    language: língua utilizada pelo nltk.tokenize para separar as sentenças
              (default='portuguese')
    regex: raw string a ser utilizada como parametro para o método split

    Retorno
    lista de sentenças
    """

    if(regex!=None):
        return re.split(regex,data)
    else:
        return tokenize.sent_tokenize(data,language=language)


def get_list_of_words(data,language='portuguese',regex=None):
    """
    Quebra o texto de entrada em uma lista de palavras
    mantendo a ordem que aparecem (pontuações também são consideradas palavras).

    Parâmetros

    data: String com texto de entrada
    language: língua utilizada pelo nltk.tokenize para separar as palavras
              (default='portuguese')
    regex: raw String a ser utilizada como parametro para o método split

    Retorno
    lista de palavras
    """

    if(regex!=None):
        return re.split(regex,data)
    else:
        return tokenize.word_tokenize(data,language=language)


def get_stem(data):
    """
    Cria radicais para as palavras

    Parâmetros

    Data: string com texto de entrada ou lista de strings

    Retorno

    String com texto com radicais ou lista com radicais
    """

    stemmer = PortugueseStemmer()

    if(type(data)==str):
        words = data.split()
        stemWords = [stemmer.stem(word) for word in words]
        stemWords = " ".join(stemWords)
    elif(type(data)==list):
        stemWords = []
        for sentence in data:
            words = sentence.split()
            stemmed = [stemmer.stem(word) for word in words]
            stemmed = " ".join(stemmed)
            stemWords.append(stemmed)
    else:
        print("Forbidden data type %s"%(type(data)))
        return ""

    return stemWords


def transform_tfidf(data, preprocessing=False, stem=False, stop_words=None, n_gram=(1,1)):
    """
    Processa dado textual e gera o modelo term frequency–inverse document frequency.

    Parametros

    data: lista de documentos a serem processados
    preprocessing: booleano que indica se os documentos serão preprocessados antes
                   da criação do modelo tf-idf
                   As etapas de preprocessamento incluem (nesta ordem)
                   - converter caracteres maiúsculos para minúsculos
                   - remoção de url, emails e pontuação
                   - remoção de stop words
                   - stemming (extração do radical das palavras)

    stem: booleano que indica se a extração de radical será realizada

    stop_words: lista de stop words ou string 'english'
                (tem efeito apenas quando preprocessing=True)

    n_gram: 2-upla de int indicando minimo e máximo de n-grams a ser extraído

    Retorno
    Tupla formada por Matriz de features TF-IDF e modelo tf-idf
    """

    if preprocessing:
        documents_processed = []

        for document in data:
            data_processed = remove_emails_and_URLs(document)
            data_processed = remove_puntuaction(data_processed)
            stop_words_flag = None

            if(stop_words!=None):
                if(stop_words=='english'):
                    stop_words_flag = stop_words
                else:
                    if(type(stop_words)==list):
                        data_processed = remove_stop_words(data_processed,stop_words)

            if(stem):
                data_processed = get_stem(data_processed)

            documents_processed.append(data_processed)

        transformer = TfidfVectorizer(stop_words=stop_words_flag,
                                      smooth_idf=False,
                                      max_features=5000)
        tfidf = transformer.fit_transform(documents_processed)

    else:
        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     ngram_range=n_gram,
                                     max_features=5000)

        training_data_features = vectorizer.fit_transform(data)
        training_data_features = training_data_features.toarray()

        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(training_data_features)

    return tfidf.toarray(),transformer


class TfidfFeatureExtractor():
    """Classe para preprocessamento (opcional) e extração das features TF-IDF"""


    def __init__(self,
                 preprocessing=False,
                 stem=False,
                 stop_words=None,
                 n_gram=(1,1),
                 max_features=None):

        """
        Inicializa os atributos do objeto com os valores padrões

        Parametros
        preprocessing: booleano que seleciona etapa de preprocessamento (default=False)

        stem: booleano que seleciona se extração de radicais será realizada (default=False)

        stop_words: lista de stop words ou string 'english' (default=None)
                    (tem efeito apenas quando preprocessing=True)

        n_gram: 2-upla de int indicando minimo e máximo de n-grams a ser extraído (default=(1,1))

        max_features: int indicando número máximo de palavras no vocabulário a ser utilizado (BoW)
                      (default=5000)

        """
        self.preprocessing = preprocessing
        self.stem = stem
        self.stop_words = stop_words
        self.n_gram = n_gram
        self.max_features=max_features
        self.vectorizer = None
        self.transformer = None
        self.tfidf_feature = None
        self.stop_words_flag = None

    def preprocessing_data(self,data):
        """
        Método que realiza etapa de preprocessamento caso atributo preprocessing=True

        Parametros
        data: lista de documentos a serem processados
        """

        documents_processed = []

        if self.preprocessing:
            for document in data:
                data_processed = remove_emails_and_URLs(document)
                data_processed = remove_puntuaction(data_processed)

                if(self.stop_words!=None):
                    if(self.stop_words=='english'):
                        self.stop_words_flag = self.stop_words
                    else:
                        if(type(self.stop_words)==list):
                            data_processed = remove_stop_words(data_processed,self.stop_words)

                if(self.stem):
                    data_processed = get_stem(data_processed)

                documents_processed.append(data_processed)
        else:
            return data

        return documents_processed

    def fit(self,data):
        """
        Método para extrair feature TF-IDF e salvar objetos a serem utilizados
        no método transform.

        Parametros:

        data: lista de documentos a serem processados
        """

        documents = self.preprocessing_data(data=data)

        if self.preprocessing:
            self.transformer = TfidfVectorizer(stop_words=self.stop_words_flag,
                                               smooth_idf=False,
                                               ngram_range=self.n_gram,
                                               max_features=self.max_features)

            self.tfidf_feature = self.transformer.fit_transform(documents)

        else:
            self.vectorizer = CountVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              ngram_range=self.n_gram,
                                              max_features=self.max_features)

            training_data_features = self.vectorizer.fit_transform(documents)
            training_data_features = training_data_features.toarray()

            self.transformer = TfidfTransformer(smooth_idf=False)
            self.tfidf_feature = self.transformer.fit_transform(training_data_features)


    def transform(self,data):
        """
        Método que retorna tf-idf array para um novo conjunto de dados a partir de
        um modelo previamente treinado pelo método fit.

        Parametros
        data: lista de documentos a serem processados
        """

        if(self.transformer == None or self.tfidf_feature == None):
            print("Unable to call transform.\nPlease call fit first.")
            return None

        documents = self.preprocessing_data(data=data)

        if self.preprocessing:
            return self.transformer.transform(documents).toarray()
        else:
            data_features = self.vectorizer.transform(documents)
            data_features = data_features.toarray()
            return self.transformer.transform(data_features).toarray()


class BagOfWordsFeatureExtractor():
    """Classe para preprocessamento (opcional) e extração das features Bag of Words"""


    def __init__(self,
                 preprocessing=False,
                 stem=False,
                 stop_words=None,
                 n_gram=(1,1),
                 max_features=None):

        """
        Inicializa os atributos do objeto com os valores padrões

        Parametros
        preprocessing: booleano que seleciona etapa de preprocessamento (default=False)

        stem: booleano que seleciona se extração de radicais será realizada (default=False)

        stop_words: lista de stop words ou string 'english' (default=None)
                    (tem efeito apenas quando preprocessing=True)

        n_gram: 2-upla de int indicando minimo e máximo de n-grams a ser extraído (default=(1,1))

        max_features: int indicando número máximo de palavras no vocabulário a ser utilizado (BoW)
                      (default=5000)

        """
        self.preprocessing = preprocessing
        self.stem = stem
        self.stop_words = stop_words
        self.n_gram = n_gram
        self.max_features=max_features
        self.vectorizer = None
        self.bow_feature = None
        self.stop_words_flag = None

    def preprocessing_data(self,data):
        """
        Método que realiza etapa de preprocessamento caso atributo preprocessing=True

        Parametros
        data: lista de documentos a serem processados
        """

        documents_processed = []

        if self.preprocessing:
            for document in data:
                data_processed = remove_emails_and_URLs(document)
                data_processed = remove_puntuaction(data_processed)

                if(self.stop_words!=None):
                    if(self.stop_words=='english'):
                        self.stop_words_flag = self.stop_words
                    else:
                        if(type(self.stop_words)==list):
                            data_processed = remove_stop_words(data_processed,self.stop_words)

                if(self.stem):
                    data_processed = get_stem(data_processed)

                documents_processed.append(data_processed)
        else:
            return data

        return documents_processed

    def fit(self,data):
        """
        Método para extrair feature BoW e salvar objetos a serem utilizados
        no método transform.

        Parametros:

        data: lista de documentos a serem processados
        """

        documents = self.preprocessing_data(data=data)

        if self.preprocessing:
            self.vectorizer = CountVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=self.stop_words_flag,
                                              ngram_range=self.n_gram,
                                              max_features=self.max_features)

        else:
            self.vectorizer = CountVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words=None,
                                              ngram_range=self.n_gram,
                                              max_features=self.max_features)

        training_data_features = self.vectorizer.fit_transform(documents)
        self.bow_feature = training_data_features


    def transform(self,data):
        """
        Método que retorna BoW array para um novo conjunto de dados a partir de
        um modelo previamente treinado pelo método fit.

        Parametros
        data: lista de documentos a serem processados
        """

        if(self.vectorizer == None or self.bow_feature == None):
            print("Unable to call transform.\nPlease call fit first.")
            return None

        documents = self.preprocessing_data(data=data)

        data_features = self.vectorizer.transform(documents)
        return data_features.toarray()



def get_word2vec_similarity(model, word1, word2):
    # Retorna a similaridade entre 2 palavreas baseada em um dicionário do Word2Vec
    # Parâmetros
    # word 1
    # word 2
    # Retorna nível de similaridade (0 a 1)
    return model.similarity(word1,word2)


def get_word2vec_most_similar(model, word):
    # Retorna qual as palavras mais similares de acordo com o contexto usando o Word2vec
    # Parâmetros
    # word
    # Retorna Conjunto de palavras e similaridade em relação a palavra passada por parâmetro
    return model.most_similar(word)


def get_word2vec_dissimilar(model, word_list):
    # Retorna palavra mais diferente em relação ao contexto dentre as que estão na lista
    # Parâmetros
    # wordList
    # Retorna palavra mais diferente dentro da lista passada por parâmetro
    sentence = " ".join(word_list)
    return model.doesnt_match(sentence.split())

def get_word2vec(model, word):
    # Retorna a representação numérica (word2vec) da palavra
    # Parâmetros
    # word
    # Retorna vetor numérico que represneta a palavra
    return model[word]
