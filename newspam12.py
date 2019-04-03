# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 15:35:34 2018

@author: sky
"""

from sklearn import svm,cross_validation
import  numpy as  np
import  pandas  as  pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_excel(r'C:\Users\lenovo\Desktop\data1.xlsx')

df.head()
df_x=df["messege"]
df_y=df["class"]
cv = TfidfVectorizer(min_df=1,stop_words='english')

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=4)
x_train=x_train.astype('str')
x_test=x_test.astype('str')
x_train.head()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')

x_traincv=cv1.fit_transform(x_train)

a=x_traincv.toarray()
cv1.inverse_transform(a[0])
x_train.iloc[0]
x_testcv=cv1.transform(x_test)
#mnb = MultinomialNB()
mnb = svm.SVC()


y_train

y_train
mnb.fit(x_traincv,y_train)
y_sc=mnb.decision_function(x_testcv)


y_predict=mnb.predict(x_testcv)

a=np.array(y_test)
count=0

for i in range (len(y_predict)):
    if y_predict[i]==a[i]:
        count=count+1
        
acurecy=(count/len(y_predict))*100
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(y_test,y_predict)

tr=[]
fp=[]
cm=[]
for i in range(10,len(y_test)):
    cnf_matrix=[]
    cnf_matrix = confusion_matrix(y_test[0:i],y_predict[0:i])
    new1=(cnf_matrix[0][0]/(cnf_matrix[0][0]+cnf_matrix[1][0]))
    if(new1==np.nan):
        tr.append(0)
    else:
        tr.append(new1)
        
    new2=(cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[1][0]))
    if(new2==np.nan):
        
        fp.append(0)
    else:
        fp.append(new2)
    cm.append(cnf_matrix)     
from sklearn.metrics import roc_curve
fpr_rf_lm, tpr_rf_lm,_= roc_curve(y_test,y_sc)
import matplotlib.pyplot as plt
plt.title("ROC curv")
plt.xlabel("true positive rate")
plt.ylabel("False positive rate")
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
tr=np.sort(tr)
fp=np.sort(fp)
plt.plot(fpr_rf_lm, tpr_rf_lm)
test=["offer"]
ex_count=cv1.transform(test)
predect=mnb.predict(ex_count)
print(acurecy)