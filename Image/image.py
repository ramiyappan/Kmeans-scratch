"""         ----------------                IMPORTS                 ---------------               """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
start = time.time()

"""         ----------------               FUNCTIONS                ---------------               """

""" Function to calculate SSE """
def cal_dist(x,y):
    return(sum((x-y)**2)) #**0.5

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
        curr_cluster = new_X[new_X['clusterID']==c][new_X.columns[:-1]]
        #print(curr_cluster)
        mean = curr_cluster.mean(axis=0)
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
        """ control exits the loop if the difference btw. old & new centroids < 1
            i.e., if the centroids move by a very little value, then we stop k-means. """
        if (abs(np.array(new_centroids) - np.array(centroids))<1).all():
            break
        else:
            """ make new_centroids as the current centroids otherwise. """
            centroids = new_centroids
    #print("break at i =", i)
    return new_df, centroids

""" Function to plot K vs WCSS """
def elbow_plot(df):
    WCSS = []
    no_cluster = np.arange(2,22,2)
    #mean = df.mean(axis=0)
    for i in no_cluster:
        #print("\nNumber of clusters = ",i)
        model,centroids = K_means(df,i)
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


"""                           ALL FUNCTIONS ARE SAME AS IN Iris.py                         """
""" CHANGE ITERATIONS COUNT IN K_MEANS() IF NEEDED.. GIVEN IT AS 100 ITERATIONS BY DEFAULT """
""" IT TAKES 28 MINS APPROX. TO RUN THIS WAY.. TRY CHANGING ITERATIONS, N_COMPONENTS, ETC.."""
"""    PLOTS MENTIONED IN THE REPORT ARE COMMENTED DOWN BELOW. CHECK IN CASE IF REQUIRED   """

"""         ----------------            GET DATA & PRE-PROCESSING           ---------------          """

df = np.array(pd.read_csv("image_test.txt",header=None, sep=","))
#df = df1[:1000,:]
#scaling = StandardScaler()
#scaling.fit(df1)
#df = scaling.transform(df1)

"""         ---------------             Applying PCA using SkLearn          ---------------          """
pca = PCA(n_components=280)
pca.fit(df)
data = pca.transform(df)
#print(data)

K = 10 # considering 10 centroids
print("Clustering input data...")
model, centroids = K_means(data,K) #calling K_means function for clustering input data based on K centroids.

clusters = model['clusterID']+1
print("clusters:")
print(clusters)


"""         ---------------              UNCOMMENT TO CHECK PLOTS           ----------------         """

""" checked elbow plot for k=2-20 and commented it. """
#elbow_plot(data)

"""
    Plot to choose the number of components that contributes to maximum variance in the dataset [eg: 95% Here]
    Tested with this plot and chose 280 components since it explains 95% of the variance.
    Commenting these plotting line of codes, after saving the plot & choosing the right amount of components needed.
"""

'''
plt.rcParams["figure.figsize"] = (12,10)
fig, ax = plt.subplots()
var = np.cumsum(pca.explained_variance_ratio_)*100
plt.plot(var, color='b')

plt.xlabel('Number of Components')
plt.ylabel('Cumulative variance (%)')
plt.title('Number of Components needed to explain Maximum variance')

x_bar = np.arange(0,351,step=10)
y_bar = var[0:351:10]
plt.bar(x=x_bar, height = y_bar)

plt.axhline(y=95, color='g', linestyle='dashed')
plt.text(55, 90, '95% Variance', color = 'red', fontsize=14)
plt.plot(280,95,marker='X',color='g',markersize=15)
plt.axvline(x=280, color='g', linestyle='dashed')
plt.text(285,97, 'x = 280', color = 'red', fontsize=10)
#ax.grid(axis='x')
ax.grid(axis='y')
plt.show()
'''

"""         Plotting just first 2 dimensions to visualize clustering.       """
"""
color_dict = dict({0:'brown',
                    1:'green',
                    2: 'orange',
                    3: 'red',
                    4: 'dodgerblue',
                    5: 'yellow',
                    6: 'grey',
                    7: 'black',
                    8: 'pink',
                    9: 'violet'})

sns.scatterplot(x=np.array(centroids)[:,0],y=np.array(centroids)[:,1],markers='x',color='r')
sns.scatterplot(x=data[:,0],y=data[:,1], hue=assigned_centroids, palette=color_dict, legend='full')
plt.show()
"""

"""         ----------------         Writing cluster IDs in a file         --------------           """

fp = open("image_testout.txt",'w')
for i in clusters:
    fp.write(str(i)+'\n')

print("finished in %.2f seconds.."%(time.time() - start))
