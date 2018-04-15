
# coding: utf-8

# In[162]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
import itertools


# In[163]:


df = pd.read_csv("/home/elodin/S22_Fake-News-Detection-Using-Natural-Language-Processing/fake_or_real_news.csv",encoding = "ISO-8859-1")
df.title.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.text.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df.columns = ['#words','title','text','label']
df.dropna(how='any')
df.head()


adf=df.copy()
for i in range(6335):
    if adf.loc[i,'label'] == 'REAL':
        adf.loc[i,'label'] = 1
    else:
        adf.loc[i,'label'] = 0
adf.head()


# In[164]:


X = adf['text']
y = adf['label']
print(X.head(),y.head())
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=0)
print(X_train.head(),X_test.head(),y_train.head(),y_test.head())


# In[165]:


tfidf_vect = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vect.fit_transform(X_train,y_train)
tfidf_test = tfidf_vect.transform(X_test)


# In[166]:


count_vect= CountVectorizer()
count_train = count_vect.fit_transform(X_train,y_train)
count_test = count_vect.transform(X_test)


# In[167]:


n_vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
n_train = n_vect.fit_transform(X_train)
n_test = n_vect.transform(X_test)


# In[168]:


vectorizer = [tfidf_vect,count_vect,n_vect]
vectorizer_train = [tfidf_train,count_train,n_train]
vectorizer_test = [tfidf_test,count_test,n_test]
zip_vect = zip(vectorizer_train,vectorizer_test,vectorizer)


# In[169]:


def NaiveBayes(nlp_train,y_train,nlp_test,y_test):
    clf = MultinomialNB(alpha=0.5, fit_prior=True)
    clf.fit(nlp_train,y_train)
    sc1 = clf.score(nlp_test,y_test)
    print("The Score is: ")
    print(sc1)
    pred = clf.predict(nlp_test)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


# In[170]:


def Logreg(nlp_train,y_train,nlp_test,y_test,nlp_vect):
    i=1
    logreg = LogisticRegression(C=9)
    logreg.fit(nlp_train,y_train)
    sc = logreg.score(nlp_test,y_test)
    print("The Score is: ")    
    print(sc)
    pred = logreg.predict(nlp_test)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])
    realest_and_fakest(nlp_vect, logreg, n=10)


# In[171]:


def RForest(nlp_train,y_train,nlp_test,y_test):
    clf1 = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=25)
    clf1.fit(nlp_train,y_train)
    sc2 = clf1.score(nlp_test,y_test)
    print("The Score is: ")    
    print(sc2)
    pred = clf1.predict(nlp_test)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


# In[172]:


def VectMachine(nlp_train,y_train,nlp_test,y_test):
    clf3 = SVC(C=100, gamma=0.1)
    clf3.fit(nlp_train, y_train)
    sc3 = clf3.score(nlp_test,y_test)
    print("The Score is: ")
    print(sc3)
    pred = clf3.predict(nlp_test)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


# In[173]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                         cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[174]:


def realest_and_fakest(vectorizer, classifier, n):

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    plt.figure()
    x_pos=[]
    y_pos=[]
    for coef, feat in (topn_class1):
        x_pos.append(feat)
        y_pos.append(coef)

    #y_pos = classifier.coef_[0][:n]
    #x_pos = feature_names[:n]
    plt.bar(x_pos, y_pos, align='center', alpha=0.5, width=0.3, color='red')
    plt.xticks(x_pos, rotation=45)
    plt.ylabel('coeff')
    plt.xlabel('top fakest words')
    plt.title('FAKE WORDS')
    #plt.ylim(-12,-8)
    #print(y_pos)
    
    #print()
    x_pos1=[]
    y_pos1=[]
    for coef, feat in reversed(topn_class2):
        x_pos1.append(feat)
        y_pos1.append(coef)
    plt.figure()
    plt.bar(x_pos1, y_pos1, align='center', alpha=0.5, width=0.3, color='blue')
    plt.xticks(x_pos1, rotation=45)
    plt.ylabel('coeff')
    plt.xlabel('top realest words')
    plt.title('REAL WORDS')


# In[175]:


for vect_train,vect_test,vect in zip_vect:
   
    if(vect == tfidf_vect):
        print("For TFIDF")
    elif(vect == count_vect):
        print("For Count")
    else:
        print("Ngrams")
    
    print("For Multinomial Naive Bayes Model")
    NaiveBayes(vect_train,y_train,vect_test,y_test)
    
    print("For Random Forest Classifiers")   
    RForest(vect_train,y_train,vect_test,y_test)
    
    print("For Support Vector Machine_Radial Basis Function Classifier")
    VectMachine(vect_train,y_train,vect_test,y_test)
    
    print("For Logarithamic Classifier")
    Logreg(vect_train,y_train,vect_test,y_test,vect)

