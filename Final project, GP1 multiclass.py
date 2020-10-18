#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import figure
import pylab
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


# In[2]:


#INTRODUCTION/EDA:
df = pd.read_csv(r"C:\Users\weizi\Desktop\IE 517\final\MLF_GP1_CreditScore.csv")


# In[3]:


df.head()


# In[ ]:





# In[4]:


df.shape


# In[5]:


# 1.1 Generate descriptive statistics:
descriptive_stat = df.describe()


# In[6]:


#1.2 Scatterplot matrix (visualizing the pair-wise correlation between
#different features):

cols = ['Sales/Revenues','Gross Margin','EBITDA Margin','Net Income Before Extras', 'Total Debt', 'Net Debt', 'Cash']
sns.pairplot(df[cols], height = 2.5)

#minimize the white space
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
plt.show()


# In[7]:


fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot(df['CFO'], dist="norm", plot=pylab)
ax.set_title("QQ/probplot for attribute CFO")
pylab.show()


# In[8]:


plt.hist(df['Cash'],color='blue',ec='black')
plt.title('Cash Histogram')
plt.xlabel('Cash')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[9]:


#Multiclass classification:

#Define credit score features
features_Credit_Score = df.iloc[:,0:26]

#Define investment ratings
target_Investment_Ratings = df.iloc[:,27]
target_Investment_Ratings


# In[10]:


#Convert Moody's rating strings into numbers:

ratings_num = {'A1':1, 'A2':2, 'A3':3,'Aa2':4,'Aa3':5, 'Aaa':6, 'B1':7,
               'B2':8, 'B3':9, 'Ba1':10,'Ba2':11, 'Ba3':12,'Baa1':13, 
              'Baa2':14, 'Baa3':15, 'Caa1':16}

#map the number rating to the string rating
rating = df['Rating'].map(ratings_num)
rating


# In[11]:


# multiclass classification for Investment Ratings


X_train, X_test, y_train, y_test = train_test_split(features_Credit_Score, 
                                                    rating,
                                                    test_size=0.2, random_state=1)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


X_train


# In[13]:


X_test


# In[14]:


y_train


# In[15]:


y_test


# In[23]:


# explained variance plot befor PCA
cov_matrix=np.cov(X_train.T)
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

#print(eigenvalues)
sorted(eigenvalues,reverse=True)

tot = sum(eigenvalues)

part=0
dimension=0

# 95% is set for principal components choice
while part/tot<0.95:
    part=part+eigenvalues[dimension]
    dimension=dimension+1
percentage=part/tot
print("we need",dimension,"components to get 95% or above of the information")


# In[24]:


var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(10,5))
plt.bar(range(1, 27), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1, 27), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#PCA transformation
pca=PCA(n_components=15)

X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)

# we only take 15 components for training
#X_train, X_test is already transformed


# In[18]:


# model A: KNN
print("KNN classification")

knn=KNeighborsClassifier(n_neighbors = 1, p = 2, metric='minkowski', algorithm='auto')

parameters={"n_neighbors":range(1,25)}

#Use bagging to reduce overfitting by drawing random combinations
#of the training set with repetition

bag = BaggingClassifier(base_estimator=tree, n_estimators=500,max_samples=1.0, 
                        max_features=1.0, bootstrap=True, 
                        bootstrap_features=False, n_jobs=1, 
                        random_state=1)


knn_train = bag.predict(X_train)
knn_test = bag.predict(X_test)



#knn_grid=GridSearchCV(knn,parameters,cv=10)

#knn_grid.fit(X_train,y_train)

print("the best n_neighbors is",knn_grid.best_params_["n_neighbors"])

print("in_sample accuracy is",knn_grid.best_score_)

#y_predict=knn_grid.predict(X_test)

print("out_of_sample accuracy is",accuracy_score(y_test,y_predict))

print("\n")


# In[ ]:


# model B: Decision Tree
print("Decision Tree classification")
from sklearn.tree import DecisionTreeClassifier

parameters={"criterion":["gini","entropy"],"max_depth":range(1,21)}

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None,
                              random_state=1)






tree_grid = GridSearchCV(tree,parameters,cv=10)

tree_grid.fit(X_train,y_train)

print("the best parameters are",tree_grid.best_params_)

print("in_sample accuracy is",tree_grid.best_score_)

y_predict = tree_grid.predict(X_test)

print("the out_of_sample accurancy is",accuracy_score(y_test,y_predict))

print("\n")


# In[ ]:


#modle C: Logistic Regression
print("Logistic Regression")
from sklearn.linear_model import LogisticRegression

#in all solvers, only liblinear support L1 penalty

log_reg=LogisticRegression(solver="liblinear")

parameters={"penalty":["l1","l2"]}

log_reg_grid=GridSearchCV(log_reg,parameters,cv=10)

log_reg_grid.fit(X_train,y_train)

print("the best parameters are",log_reg_grid.best_params_)

print("in_sample accuracy is",log_reg_grid.best_score_)

y_predict = log_reg_grid.predict(X_test)

print("the out_of_sample accurancy is",accuracy_score(y_test,y_predict))

print("\n")


# In[ ]:


# ensembling

print("Random Forest")

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_jobs=-1)
parameters={"criterion":["gini","entropy"],"n_estimators":[10,25,50,100,250,500,750,1000]}

forest_grid=GridSearchCV(forest,parameters,cv=10)

forest_grid.fit(X_train,y_train)

print("the best parameters are",forest_grid.best_params_)

print("in_sample accuracy is",forest_grid.best_score_)

y_predict = forest_grid.predict(X_test)

print("the out_of_sample accurancy is",accuracy_score(y_test,y_predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




