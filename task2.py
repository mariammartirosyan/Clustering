import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sb
from sklearn.decomposition import PCA

#part a
#read the data
dataFrame = pd.read_csv("seeds.csv")
data = dataFrame.iloc[:, :-1]
labels = dataFrame['Type']

#normalize the data
#source: https://scikit-learn.org/stable/modules/preprocessing.html
normalizedData = preprocessing.StandardScaler().fit(data).transform(data)

#perform PCA via SVD
u, sigma, v = np.linalg.svd(normalizedData, full_matrices=False)
# source that helped me: https://www.samlau.me/test-textbook/ch/19/pca_svd.html
#take the first two principal components
svdPC = v[:2].T
#project data onto the first two principal components
projectedDataSVD = -1 * normalizedData @ svdPC
#source end
print(f"Projected data with PCA via SVD\n{projectedDataSVD}")

#perform PCA
pca = PCA(n_components=2, svd_solver='full')
#project data onto the first two principal components
projectedDataPCA = pca.fit_transform(normalizedData)
print(f"Projected data with PCA\n{projectedDataPCA}")

if(np.allclose(projectedDataSVD,projectedDataPCA)):
    print("Porjected data is the same.")

#part b
figure = plt.figure()
figure.subplots_adjust(hspace=0.5, wspace=0.5)

#plot the results from PCA via SVD
projectedDataSVD = np.concatenate((projectedDataSVD, np.array(labels[:, None])), axis=1)
projectedDataSVD_DF = pd.DataFrame(data=projectedDataSVD, columns=['PC1', 'PC2', 'Type'])
figure.add_subplot(1, 2, 1)
plt.title("PCA via SVD")
sb.scatterplot(data=projectedDataSVD_DF, x='PC1', y='PC2', hue='Type', palette="muted")

#plot the results from PCA
projectedDataPCA = np.concatenate((projectedDataPCA, labels[:, None]), axis=1)
projectedDataPCA_DF = pd.DataFrame(data=projectedDataPCA, columns=['PC1', 'PC2', 'Type'])
figure.add_subplot(1, 2, 2)
plt.title("PCA")
sb.scatterplot(data=projectedDataPCA_DF, x='PC1', y='PC2', hue='Type', palette="muted")
plt.show()

X = np.array([[6, -1],[2, 3]])















