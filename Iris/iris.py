"""         ----------------                IMPORTS                 ---------------               """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns

"""         ----------------               FUNCTIONS                ---------------               """

""" Function to calculate SSE. """
def cal_dist(x,y):
    return(sum((x-y)**2))

""" Function to assign all data points to the nearest centroid. """
def nearest_centroids(x,c):
    assigned_centre = []
    for i in x:
        distance = []
        for j in c:
            distance.append(cal_dist(i,j))
        assigned_centre.append(np.argmin(distance))
    return assigned_centre

""" Function to calculate the mean of all points in a centroid and making it as the new centroid. """
def cal_NewCentroids(X,clusters):
    new_centroids = []
    new_X = pd.concat([pd.DataFrame(X),pd.DataFrame(clusters, columns = ['clusterID'])],axis=1)
    #print("new_X : ",new_X )
    for c in set(new_X['clusterID']):
        current_cluster = new_X[new_X['clusterID']==c][new_X.columns[:-1]]
        #print(curr_cluster)
        mean = current_cluster.mean(axis=0)
        #print(mean)
        new_centroids.append(mean)
    return new_centroids, new_X

""" K_means Function to run iterations and keep shifting the centroids depending on the mean. """
def K_means(X,K):
    """ picking k random indicies from the dataset to be the initial centroids. """
    centroids_idx = random.sample(range(0,len(X)),K) #getting random indices
    centroids = []
    for i in centroids_idx:
        centroids.append(X[i])
        #print(centroids)
    """ Starting with k random initial centroids. """
    for i in range(100):
        assigned_centroids = nearest_centroids(X,centroids)
        new_centroids,new_df = cal_NewCentroids(X,assigned_centroids)
        #print("new centroids are: ")
        #print(new_centroids)
        """ control exits the loop if the difference btw. old & new centroids < 0.001
            i.e., if the centroids move by a very little value, then we stop k-means. """
        if (abs(np.array(new_centroids) - np.array(centroids))<0.001).all():
            break
        else:
            """ make new_centroids as the current centroids otherwise. """
            centroids = new_centroids
    #print("break at i =", i)
    return new_df, centroids, assigned_centroids

""" Function to plot K vs WCSS """
def elbow_plot(df):
    WCSS = []
    no_cluster = np.arange(2,22,2)
    #mean = df.mean(axis=0)
    for i in no_cluster:
        #print("\nNumber of clusters = ",i)
        model,centroids, assigned_centroids = K_means(df,i)
        Tot_SSE = 0
        length = 0
        BSS = 0
        for j in range(len(centroids)):
            #print("for clusterID = ",j)
            WSS = 0
            current = model[model['clusterID']==j][model.columns[:-1]]
            #print("length of current cluster:")
            #print(current)
            #breakpoint()
            length += len(current)
            for row,k in current.iterrows():
                #print(k,centroids[j],cal_dist(centroids[j],k))
                #breakpoint()
                WSS += np.round(cal_dist(centroids[j],k))
            #print("WSS = ",WSS)
            #BSS += np.round(len(current) * (sum((mean - centroids[j])**2)))
            #print("BSS = ",BSS)
            Tot_SSE += WSS + BSS
            #print(centroids)
        #print("Tot_SSE = ",Tot_SSE)
        WCSS.append(Tot_SSE)
        #print("Total number of data points = ",length)
    #print("\n\nWCSS = ", WCSS)
    #print(centroids)
    #print("K = ",no_cluster)
    plt.figure()
    plt.xlim(0,22)
    x_plot = [2,4,6,8,10,12,14,16,18,20]
    plt.plot(no_cluster,WCSS,marker='o')
    plt.xticks(no_cluster,x_plot)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow plot of K vs WCSS")
    plt.show()


"""         ----------------            GET DATA & PRE-PROCESSING           ---------------          """

df = np.array(pd.read_csv("test.txt",header=None, sep=" "))

K = 3 # considering 3 centroids
print("Clustering input data...")

model, centroids, assigned_centroids = K_means(df,K) #calling K_means function for clustering input data based on K centroids.
cluster_id = model['clusterID']+1
print("cluster id's for data points :")
print(cluster_id)

"""         ----------------                    PLOTS                       ---------------          """

""" checked elbow plot for k=2-20 and commented it. """
#elbow_plot(df)

""" Plotting with any 2 features from 4 features, to just visualize clustering. """
"""
sns.scatterplot(x=np.array(centroids)[:,1],y=np.array(centroids)[:,2],s=150,color='black')
sns.scatterplot(x=df[:,1],y=df[:,2], hue=assigned_centroids)
plt.show()
"""

"""         ----------------         Writing cluster IDs in a file         --------------           """

fp = open("testout.txt",'w')
for i in cluster_id:
    fp.write(str(i)+'\n')