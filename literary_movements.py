#Complete Python notebook for comparing and classifying literary movements:
# Written and shared with love by Victor Saenger. saenger.v(at)gmail.com
#linkedIn https://www.linkedin.com/in/victor-saenger/
# Sept 2019.

#Tensorflow , tf-hub and tf-sentencepiece, all needed to calculate #embbedings.
#Check this discussion to install compatible versions.
#At the time of this writing, a compatible mix of packages is tf #v1.13.1,
#hub v0.5.0 and sentencepiece v0.1.82.1:

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
#numpy and random
import numpy as np
import random
# sklearn and imblearn packages:
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import EditedNearestNeighbours
#Visualization:
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

f_rom = open("Romanticism.txt", "r")
f_rea = open("Realism.txt", "r")
f_sur = open("Surrealism.txt", "r")

list_all_rom = f_rom.readlines()
list_all_rea = f_rea.readlines()
list_all_sur = f_sur.readlines()


def merge_list(list_text):
    l = len(list_text)
    c = 1
    merge = eval(list_text[0])[1:]
    while c < l:
          merge = merge + eval(list_text[c])[1:]
          c += 1
    return merge

merged_list_rom = merge_list(list_all_rom)
merged_list_rea = merge_list(list_all_rea)
merged_list_sur = merge_list(list_all_sur)
merged_list_all = merged_list_rom + merged_list_rea + merged_list_sur


# Graph set up.
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module('/home/victor/Documents/USEM/')
  # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)

#Function to compute all embeddings and distance between all possible quote pairs:
#Be patient, takes a little while
def similarity_matrix(merge_list):
    #initialize distance array:
    similarity_matrix = np.zeros([len(merge_list), len(merge_list)])
    #initialize embeddings array:
    emb_all = np.zeros([len(merge_list),512])
    #Outer for loop:
    for i in range(0,len(merge_list)):
        #Here is where we run the previously started session, so it is important to run previous step succesfully:
        i_emb = session.run(embedded_text, feed_dict={text_input: [merge_list[i]]})
        emb_all[i,:] = i_emb
        #Inner for loop
        for j in range(0,len(merge_list)):
            j_emb = session.run(embedded_text, feed_dict={text_input: [merge_list[j]]})
            # print(j)
            similarity_matrix[i,j] = np.inner(i_emb,j_emb)
    return similarity_matrix, emb_all

#Compute embeddings and distances:
sM_All,e_All = similarity_matrix(merged_list_all)

classes = np.asarray([1 for i in range(len(merged_list_rom))] + \
[2 for i in range(len(merged_list_rea))] + [3 for i in range(len(merged_list_sur))])


#This pipeline allow us to change the type of classifier as clf is an input:
def class_pipeline(features, class_ground, clf):
    #Call Shuffle splitter:
    ss = StratifiedShuffleSplit(n_splits=50)

    #sm = EditedNearestNeighbours()
    #[features, class_ground] = sm.fit_resample(features, class_ground)

    #Alocate variables:
    class_pred_all = []
    class_test_all = []
    f1_score_all = []

    #K fold cross validation shuffled:
    for train, test in ss.split(features,class_ground):
        print(train)

        #train/test split embedings:
        features_train = features[train,:]
        class_train = class_ground[train]
        features_test = features[test,:]
        class_test = class_ground[test]

        #Fit the model (clf is a typically used word to call any classifier):
        clf.fit(features_train,class_train)
        #Predict literature class from word embeddings:
        class_predicted = clf.predict(features_test)
        # Calculate F1 score and store all predicted and test classes:
        f1_score_all = np.append(f1_score_all,f1_score(class_test,class_predicted,average='weighted'))
        class_pred_all = np.append(class_pred_all,class_predicted)
        class_test_all = np.append(class_test_all,class_test)

    return class_pred_all, class_test_all, f1_score_all

#define classifier and run cross-validation:
clf = MLPClassifier(max_iter=500,activation="tanh")
class_pred, class_test, f1_score = class_pipeline(StandardScaler().fit_transform(e_All),classes,clf)

plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(class_test, class_pred), \
annot=True,linewidths=2, cmap="YlGnBu",fmt="g", \
xticklabels=['Romanticism','Realism','Surrealism'],\
yticklabels=['Romanticism','Realism','Surrealism'])

#to plot quotes heatmap:
plt.figure(figsize=(18, 16))
sns.heatmap(data=sM_All,annot=False,linewidths=0.1, cmap="YlGnBu")
plt.show()

#This code is to find most similar quote from a given author:
l_all = list_all_rom + list_all_rea + list_all_sur

#merge all names and repeat names as many times as quotes per author:
names_mult = []
for i in l_all:
    names_mult = names_mult + [eval(i)[0]] * int(len(eval(i))-1)

#Find most similar quote from author function:

def find_closest(sM,merged_list,names_m,author):
    close = np.argsort(sM[author,:])[len(sM)-2]
    return names_mult[author], merged_list[author], names_m[close], merged_list[close] 

#Test the function. Last input is the author you wish to test as given in names_mult:
#Here 0-7 is Poe for example.
find_closest(sM_All,merged_list_all,names_mult,0)


#to render graph:
#Import array to netowrkx object with a given threshold:
G = nx.from_numpy_array(sM_All>0.25)
#figure layout and size:
pos = nx.kamada_kawai_layout(G)
plt.figure(figsize=(10, 10))
plt.axis('off')
#node size input should be changed according to network size:
nx.draw_networkx_nodes(G, pos, node_size=100, \
#this simple trick allows to render the graph with colors according to classes:
cmap=plt.cm.Dark2, node_color = list(classes))
nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.2)
plt.show()
