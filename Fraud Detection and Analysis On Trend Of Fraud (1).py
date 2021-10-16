#!/usr/bin/env python
# coding: utf-8

# #  A PROJECT ON FRAUD DETECTION AND ANALYSIS ON TREND OF FRAUD

# ## Importing Required Libraries

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Loading Dataset

# In[4]:


df=pd.read_csv("Fraud_detection.csv")


# ### Shape of Dataset

# In[5]:


print(df.shape)


# In[4]:


print(df.head(10))


# In[5]:


print(df.describe())


# In[6]:


df.info()


# In[7]:


df.corr()


# In[8]:


print(df.groupby('type').size()) 


# In[9]:


print(df.groupby('isFraud').size()) 


# ### Checking For Null Values

# In[10]:


df.isnull().sum()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.isnull().sum()


# ### Deleting Not Required Coloumns

# In[13]:


df_new=df.drop(["nameOrig","nameDest","isFlaggedFraud"],axis=1)


# In[14]:


df_new


# ## Data Visualization

# In[15]:


sns.heatmap(df_new.corr(),annot=True)


# In[16]:


df_new.plot(kind='box', subplots= True, layout=(8,1),sharex = False, sharey= False,figsize=(10,20))
plt.show()


# In[17]:


df_new.hist(figsize=(10,10))
plt.show()


# In[18]:


from pandas.plotting import scatter_matrix
scatter_matrix(df_new)
plt.gcf().set_size_inches((20, 20)) 
plt.show()


# In[19]:


df_new.drop(['newbalanceDest','oldbalanceOrg'],inplace=True,axis=1)


# ## String Values to Numbers

# In[20]:


DF=df_new.replace("PAYMENT",0)
DF1=DF.replace("CASH_IN",1)
df=DF1.replace("CASH_OUT",2)
df1_new=df.replace("TRANSFER",3)
df2_new=df1_new.replace("PAYMENT",4)
df_new=df2_new.replace("DEBIT",5)


# In[21]:


df_new.head()


# In[22]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]


# In[23]:


plt.figure(figsize=(10,8))
bx =sns.boxplot(x= "amount",data = X )
bx.set_title("Boxplot")
plt.show()


# In[24]:


fig,ax=plt.subplots()
print(fig)
print(ax)
ax.violinplot(X["step"],vert=False)
plt.show()


# # DATA TRAINING

# In[25]:


from sklearn.preprocessing import MinMaxScaler


# In[26]:


scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(X)
rescaledX


# In[27]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# ## LOGISTIC REGRESSION

# In[29]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
logit=linear_model.LogisticRegression(C=1e10,max_iter=1e5)
logit.fit(X_train,Y_train)


# In[30]:


print('Logistic Regression Accuracy Score on Test data:', logit.score(X_test,Y_test))
print('Logistic Regression Accuracy Score on train data:', logit.score(X_train,Y_train))


# In[31]:


from sklearn.metrics import confusion_matrix
predictions = logit.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# ## SVM CLASSIFIER

# In[32]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


X=df_new[["type","amount"]].values
Y=df_new['isFraud'].values
X,Y


# In[34]:


sc = StandardScaler()
X = sc.fit_transform(X)


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)


# In[36]:


svm =SVC(kernel="poly")
svm.fit(x_train,y_train)


# In[37]:


y_pred = svm.predict(x_test)
accuracy_score(y_test,y_pred)


# In[38]:


from matplotlib.colors import ListedColormap


# In[39]:


setX, setY = x_train, y_train
start = setX[:,0].min() - 1
stop = setX[:,0].max() + 1
x1 = np.arange(start, stop, step=0.01)
start = setX[:,1].min() - 1
stop = setX[:,1].max() + 1
x2 = np.arange(start, stop, step=0.01)
xx,yy = np.meshgrid(x1,x2)


# In[40]:


z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
z


# ## POLY KERNEL

# In[41]:


plt.contourf(xx, yy, z, alpha = 0.75, cmap = ListedColormap(('green','red')))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(),yy.max())
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.show()


# In[42]:


svm =SVC(kernel="rbf")
svm.fit(x_train,y_train)


# In[43]:


y_pred = svm.predict(x_test)
accuracy_score(y_test,y_pred)


# In[44]:


z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
z


# ## RBF KERNEL

# In[45]:


plt.contourf(xx, yy, z, alpha = 0.75, cmap = ListedColormap(('green','red')))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(),yy.max())
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
plt.show()


# In[46]:


from sklearn.naive_bayes import MultinomialNB,  GaussianNB, BernoulliNB
from sklearn import metrics
from sklearn.svm import LinearSVC
svm =LinearSVC()
svm.fit(x_train,y_train)


# In[47]:


y_pred = svm.predict(x_test)
accuracy_score(y_test,y_pred)


# In[48]:


z = svm.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
z


# ## LINEARSVC

# In[49]:


sns.set(rc={"figure.figsize":(7,5)})
plt.contourf(xx, yy, z, alpha = 0.75, cmap = ListedColormap(('green','red')))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(),yy.max())
plt.scatter(x_train[:,0],x_train[:,1],c=y_train)

plt.show()


# In[50]:


accuracy_score(y_test,y_pred)


# ## DECISION TREE CLASSIFIER

# In[51]:


from sklearn.tree import DecisionTreeClassifier


# In[52]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[53]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
clf = DecisionTreeClassifier()
clf.fit(x_test,y_test)


# In[54]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[55]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,Y_train)
print('Decision Tree Accuracy Score:', dt_clf.score(X_test,Y_test))


# In[56]:


from sklearn.metrics import confusion_matrix
predictions = dt_clf.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# ## NAIVE BAYES CLASSIFIER

# ## GaussianNB

# In[57]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
from sklearn import metrics 
from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb.fit(X_train,Y_train)
print('GaussianNB Accuracy Score:', gnb.score(X_test,Y_test))


# In[58]:


from sklearn.metrics import confusion_matrix
predictions = gnb.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# ## BernoulliNB

# In[59]:


from sklearn.naive_bayes import BernoulliNB
bnb =BernoulliNB()
bnb.fit(X_train, Y_train)
#print(metrics.accuracy_score(X_test,Y_test))
y_pred_class = bnb.predict(X_test)
print(metrics.accuracy_score(Y_test, y_pred_class))


# In[60]:


from sklearn.metrics import confusion_matrix
predictions = bnb.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# ## MultinomialNB

# In[61]:


from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()
mnb.fit(X_train,Y_train)
print('GaussianNB Accuracy Score:', mnb.score(X_test,Y_test))


# In[62]:


from sklearn.metrics import confusion_matrix
predictions = mnb.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# ## RANDOM FOREST CLASSIFIER

# In[63]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(class_weight='balanced')
rf_clf.fit(X_train,Y_train)
print('Random Forest Classifier Accuracy Score:', rf_clf.score(X_test,Y_test))


# In[64]:


from sklearn.metrics import confusion_matrix
predictions = rf_clf.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# In[65]:


rf_clf.fit(x_train,y_train)


# ## K-Neighbors Classifier

# In[66]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
from sklearn.neighbors import KNeighborsClassifier
knn_clf= KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
knn_clf.fit(X_train,Y_train)
print('KNN Accuracy Score:', knn_clf.score(X_test,Y_test))


# In[67]:


from sklearn.metrics import confusion_matrix
predictions = knn_clf.predict(X_test)
print(confusion_matrix(Y_test,predictions))


# In[68]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


rmse_val =[]
for K in range(20):
        K =K+1
        model= neighbors.KNeighborsRegressor(n_neighbors = K)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        error=sqrt(mean_squared_error(Y_test,pred))
        rmse_val.append(error)
        print(K, error)


# In[70]:


curve = pd.DataFrame(rmse_val)
curve.plot()


# In[71]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[72]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[4,5,6,7,8,9,10,11,12,13]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,Y_train)
model.best_params_


# In[73]:


X = np.array(X)
X


# In[74]:


kmeans = KMeans(n_clusters = 13, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[75]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'black', label = 'fraud')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red', label = 'not fraud')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# ## ANN  DEEP LEARNING

# In[76]:


from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[77]:


X=df_new.drop(["isFraud"],axis=1)
Y=df_new["isFraud"]
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[78]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[79]:


model.fit(X,Y, epochs=50, batch_size=10)


# In[80]:


_, accuracy = model.evaluate(X,Y)
print('ANN ModelAccuracy: %.2f'%(accuracy*100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## PAIRPLOTS

# In[81]:


sns.pairplot(X_train[["step","type","amount","newbalanceOrig","oldbalanceDest"]],diag_kind='kde',hue='type')


# In[82]:


sns.pairplot(X_train[["step","type","amount","newbalanceOrig","oldbalanceDest"]])
sns.set(rc={"figure.figsize":(20,20)})


# ## PREDICTING VALUES

# In[83]:


logit.predict([[1,2,10000,16296.36,0.0]]) # Logistic Regression


# In[84]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))


# In[85]:


rf_clf.predict([[1,2,10000,16296.36,0.0]])  # Random Forest Classifier 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




