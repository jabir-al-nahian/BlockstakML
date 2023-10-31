#!/usr/bin/env python
# coding: utf-8

# In[184]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[185]:


#Read data from local
df = pd.read_csv(r"C:\Users\jabir\OneDrive\Desktop\ML Assessment Dataset (Bank Data) - Sheet1.csv")


# In[186]:


#Print the concise summery of the dataset
df.info()


# In[187]:


print("Tota Rows and Columns (Rows, Columns): ",df.shape)


# In[188]:


#Print first ten rows of the dataset
df.head(10)


# In[189]:


#Client Subscription Ratio
df['y'].value_counts()


# In[190]:


#Bar Chart of Subscription Result
import plotly.express as px
cut_counts = df['y'].value_counts()
fig = px.bar(x=cut_counts.index, y=cut_counts.values)
fig.show()


# In[194]:


#Checking Null Value
df.isna().sum()


# In[195]:


df['age'].value_counts()


# In[197]:


#Data labeling 
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome','y']

#Initialize LabelEncoder
label_encoder = LabelEncoder()

#Encode categorical variables
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


# In[198]:


df.head()


# In[199]:


df.describe()


# In[207]:


data_corr = df.corr()


# In[208]:


data_corr


# In[209]:


plt.figure(figsize=(10,10))
sns.heatmap(data_corr, annot= True, linewidth = 0.5, fmt='0.2f', cmap='YlGnBu')


# In[210]:


#Split data into train and testing
from sklearn.model_selection import train_test_split
X = df.drop('y', axis=1)
Y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[211]:


#Accuracy calculate Function
total_accuracy = {}
def accuracy(model):
    pred = model.predict(X_test)
    accu = metrics.accuracy_score(y_test,pred)
    print("Acuuracy Of the Model: ",accu,"\n")
    total_accuracy[str((str(model).split('(')[0]))] = accu


# In[212]:


#Confusion Matrix Function
from sklearn import metrics
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
    plt.figure(1)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['YES','NO']
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[213]:


#ROC Curve Function
from sklearn.metrics import roc_curve, auc
def report_performance(model):

    model_test = model.predict(X_test)

    print("Confusion Matrix:")
    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))
    print("\nClassification Report: ")
    print(metrics.classification_report(y_test, model_test))
    #cm = metrics.confusion_matrix(y_test, model_test)
    plot_confusion_metrix(y_test, model_test)

total_fpr = {}
total_tpr = {}
def roc_curves(model):
    predictions_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(predictions_test,y_test)
    roc_auc = auc(fpr, tpr)
    total_fpr[str((str(model).split('(')[0]))] = fpr
    total_tpr[str((str(model).split('(')[0]))] = tpr
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[214]:


#Classify using Decision Tree
from sklearn.tree import DecisionTreeClassifier
dctmodel = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=0)
dctmodel.fit(X_train, y_train.ravel())
accuracy(dctmodel)
report_performance(dctmodel) 
roc_curves(dctmodel)


# In[215]:


#Classify using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train.ravel())
accuracy(naive_bayes)
report_performance(naive_bayes) 
roc_curves(naive_bayes)


# In[216]:


for i in total_fpr.keys():
    plt.plot(total_fpr[i],total_tpr[i],lw=1, label=i)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend()

