import string
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import re
import scipy

def isinclude(text, sub_text):
    if isinstance(text, str):
        return sub_text.lower() in text.lower()
    else:
        return False

def preprocess(names_list):
    names_list = names_list.astype(str)
    names_list = names_list.apply(lambda x: str(x).lower())
    names_list = names_list.apply(lambda x: x.strip())

    translator = str.maketrans("", "", string.punctuation)
    names_list = names_list.apply(lambda x: str(x).translate(translator))

    names_list = names_list.apply(lambda x: ''.join(c for c in x if not unicodedata.combining(c)))

    return names_list

def most_common(df, k):
    df = pd.Series(df)
    cloud = ''
    for sentence in tqdm(df.values):
        cloud = str(cloud) + str(sentence) + " "
    
    counter = Counter(cloud.split())
    most_occur = counter.most_common(k)
    print(most_occur)
    most_occur = [pair[0] for pair in most_occur]
    return most_occur

def extract_sw_sent(sentence, sw):
    new_sent = ""
    for word in sentence.split():
        if word not in sw:
            new_sent += word + ' '
    return new_sent

def extract_sw(df, sw):
    df = df.apply(lambda x: extract_sw_sent(x, sw))
    df = df.apply(lambda x: x.strip())
    return df

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def transform_tqdm(serie, vectorizer, chunk_size = 10_000):
    print(len(serie))
    number_chunks = len(serie) // chunk_size
    tf_idf_matrix = scipy.sparse.csr.csr_matrix([])
    for c, chunk in tqdm(enumerate(np.array_split(serie, number_chunks))):
        if not c:
            tf_idf_matrix = vectorizer.transform(chunk)
        else:
            tf_idf_matrix_part = vectorizer.transform(chunk)
            tf_idf_matrix = scipy.sparse.vstack((tf_idf_matrix, tf_idf_matrix_part))
    
    return tf_idf_matrix

def search_tqdm(serie, model, chunk_size, k):
    number_chunks = serie.shape[0] // chunk_size
    last_idx = serie.shape[0] % chunk_size

    distances_search_list = list()
    indexes_search_list = list()

    for ii in tqdm(range(number_chunks)):
        search_result = model.kneighbors(serie[ii*chunk_size:(ii+1)*chunk_size], 5, return_distance=True)
        distances_list_chunk, indexes_list_chunk = np.array(search_result)[:,:,:]

        distances_search_list = [*distances_search_list,*distances_list_chunk]
        indexes_search_list = [*indexes_search_list,*indexes_list_chunk]
    
    if last_idx:
        search_result = model.kneighbors(serie[-last_idx:], 5, return_distance=True)
        distances_list_chunk, indexes_list_chunk = np.array(search_result)[:,:,:]

        distances_search_list = [*distances_search_list,*distances_list_chunk]
        indexes_search_list = [*indexes_search_list,*indexes_list_chunk]
    
    return distances_search_list, indexes_search_list