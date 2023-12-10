from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, auc
from itertools import cycle
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

class Split:
    def __init__(self,data) -> None:
        self.data=data
        self.models=None
    #split to X y
   
    def split_to_x_y(self):
        X= self.data.drop(['Heart Attack Risk'], axis=1).values
        y = self.data['Heart Attack Risk'].values
        return X,y
    #Rescaling  the values
   
    def rescale(self,X,y):
        scaler =StandardScaler()
        smote=SMOTE(random_state=30)
        #pca=PCA(n_components=15)
        X_rescaled, y_rescaled =smote.fit_resample(X,y)
        #X_pca=pca.fit_transform(X_rescaled)
        X_scaled=scaler.fit_transform(X_rescaled)
        return X_scaled,y_rescaled
   
    def split(self,X,y):
        #split into train annd test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        #rescaling y_train, y_test
        y_train=np.ravel(y_train)
        y_test=np.ravel(y_test)
        return X_train,X_test,y_train,y_test
    
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
    
class ModelAnalysis:
    def __init__(self,X_train,y_train,X_test,y_test) -> None:
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test



    def roc_auc_plot(self,m_name,y_test,y_pred):
       fpr = dict()
       tpr = dict()
       roc_auc = dict()
       
       for i in range(2):
         fpr[i], tpr[i], _ = roc_curve(y_test, y_pred[:, i])
         roc_auc[i] = auc(fpr[i], tpr[i])
       
       fig,ax=plt.subplots()
       lw = 2
       for i in range(2):
         ax.plot(
         fpr[i],
         tpr[i],
         lw=lw,
         label=f"{m_name} ROC curve (area = %0.2f)" % roc_auc[i],
          )
       ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
       ax.set_xlim([0.0, 1.0])
       ax.set_ylim([0.0, 1.05])
       ax.set_xlabel("False Positive Rate")
       ax.set_ylabel("True Positive Rate")
       ax.set_title("Receiver operating characteristic example")
       ax.legend(loc="lower right")
       st.pyplot(fig)
       #return roc_auc[0]
    

    
    def summarize_models(self):
        self.models=[
            ['Logistic Regresion',BaggingClassifier(LogisticRegression(), n_estimators=100,random_state=20 )],
            ['Decision Tree Classifier',BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=20)],
            ['Random Forest Classifier',RandomForestClassifier(n_estimators=100)],
            ['Support Vector Machine',BaggingClassifier(SVC(kernel='rbf',probability=True),n_estimators=10, random_state=20)],
            ['Naive Bayes',GaussianNB()],
            ['K Nearest Neighbor',KNeighborsClassifier(n_neighbors=12)]
             ]
        result=[]

        
        for model in self.models:
            m_name=model[0]
            m_model=model[1]
            m_model.fit(self.X_train,self.y_train)
            y_pred=m_model.predict(self.X_test)
            y_pred_proba=m_model.predict_proba(self.X_test)
            #print(y_pred)
            accuracy=accuracy_score(self.y_test,y_pred)
            result.append({'Model': m_name , 
                           'Accuracy':accuracy,
                           ' ROC_AUC score':roc_auc_score(self.y_test,y_pred) })
            #self.roc_auc_plot(m_name,self.y_test,y_pred_proba)
            
        result_df=pd.DataFrame(result)
        return result_df    

        
 

    



    

