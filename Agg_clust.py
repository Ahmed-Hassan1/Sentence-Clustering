# Clean the data by removing non english chars, normalizing the capitalization and stemming.  
# we vectorize the sentences and add the result as a column beside each sentence.
# we cluster them then assign each cluster to a sentence
# we can export each cluster easily.

import pandas as pd
import numpy as np
import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

#Load the CSV file. The file name here
sentences = pd.read_csv('sentences.csv')

#The column's name in the CSV file
col_name='sentence'

#The trained data to base the vectoring on
#https://www.sbert.net/docs/pretrained_models.html this link contains different models
model_name='all-MiniLM-L6-v2'

model=SentenceTransformer(model_name)


sentence_vecs=model.encode(sentences[col_name])

#Normalizing the vectors to unit length
sentence_vecs = sentence_vecs / np.linalg.norm(sentence_vecs,axis=1,keepdims=True)

#You can play with the paramerters for the desired output
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering To read more
#n_clusters=None, *, affinity="euclidean", memory=None, connectivity=None, compute_full_tree="auto", linkage="ward", distance_threshold=None, compute_distances=False)
cluster_model = AgglomerativeClustering(n_clusters=None,distance_threshold=1.5)
cluster_model.fit(sentence_vecs)
cluster_assignment=cluster_model.labels_



clustered_sents={}
for sent_id,cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sents:
        clustered_sents[cluster_id]=[]

    clustered_sents[cluster_id].append(sentences[col_name][sent_id])



clust_num=[]
clusted=[]
for i,cluster in clustered_sents.items():
    clust_num.append(i)
    clusted.append(cluster)
    print("Cluster ",i+1)
    print(cluster)
    print("")

all_combined=dict(zip(clust_num,clusted))
print(all_combined)

df= pd.DataFrame.from_dict(all_combined,orient='index')
df=df.transpose()
df.to_csv('out2.csv',index=False)