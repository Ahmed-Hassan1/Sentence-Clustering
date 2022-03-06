# Clean the data by removing non english chars, normalizing the capitalization and stemming.  
# we vectorize the sentences and add the result as a column beside each sentence.
# we cluster them then assign each cluster to a sentence
# we can export each cluster easily.

import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

#Load the CSV file. The file name here
sentences = pd.read_csv('sentences.csv')

#The column's name in the CSV file
col_name='sentence'

#Number of clusters
num=5

#The trained data to base the vectoring on
model_name='bert-base-nli-mean-tokens'

model=SentenceTransformer(model_name)

#The column's name in the CSV file
sentence_vecs=model.encode(sentences[col_name])


cluster_model = KMeans(n_clusters=num)
cluster_model.fit(sentence_vecs)
cluster_assignment=cluster_model.labels_

clustered_sents=[[] for i in range(num)]
for sent_id,clu_id in enumerate(cluster_assignment):
    clustered_sents[clu_id].append(sentences[col_name][sent_id])



clust_num=[]
clusted=[]
for i,cluster in enumerate(clustered_sents):
    clust_num.append(i)
    clusted.append(cluster)
    print("Cluster ",i+1)
    print(cluster)
    print("")

all_combined=dict(zip(clust_num,clusted))
print(all_combined)

df= pd.DataFrame.from_dict(all_combined,orient='index')
df=df.transpose()
df.to_csv('out3.csv',index=False)