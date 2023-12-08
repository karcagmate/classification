from sklearn.pipeline import Pipeline, FeatureUnion,make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from TrainTestSplit import Split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Voting Classifiers:

#Hard Voting: Combine the predictions from multiple models and select the class that gets the majority of votes.
#Soft Voting: Combine the predicted probabilities from multiple models and select the class with the highest average probability.
#Bagging (Bootstrap Aggregating):

#Train multiple instances of the same classification algorithm on different subsets of the training data.
#Combine their predictions (e.g., averaging for regression, voting for classification).
#Boosting:

#Train multiple weak learners sequentially, with each one correcting the errors of the previous one.
#Popular algorithms include AdaBoost, Gradient Boosting (e.g., XGBoost, LightGBM), and CatBoost.
#Stacking:

#Train multiple models and use another model (meta-model) to combine their predictions.
#The meta-model is trained on the outputs of the base models.



class CreatePipeLine:
    def __init__(self,data) -> None:
       self.data=data

       
          
    #def rescale(self):
    #    steps=[#('smote',SMOTE(random_state=50)),
    #           ('pca',PCA()),
      #         ('standard scaler', StandardScaler())]
     #   rescale_pipeline=Pipeline(steps) 
     #   return rescale_pipeline 
    
    #def combine_classificaiton(self):
        #estimators=[(RandomForestClassifier()),
         #            (SVC(kernel='linear',probability=True)),
        
       # est_ensemble=VotingClassifier(estimators=
                                  #    [
                                   #       ('randomforest',RandomForestClassifier(n_estimators=100)),
                                    #      ('svm',SVC(kernel='linear',probability=True)),
                                     #     ('gaussian',GaussianNB())
                                     # ],voting='soft')
       # classifcation_pipe = make_pipeline(self.rescale(),est_ensemble)
       # adaboost=AdaBoostClassifier(estimator=RandomForestClassifier(n_estimators=100),n_estimators=50,random_state=0)
       # return adaboost
    def combine__classification(self,final_est):
        tree= BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,random_state=20)
        svm=BaggingClassifier(SVC(kernel='rbf',probability=True),n_estimators=10,random_state=20)
        estimatorss=[('svm',svm),('tree',tree),('random forest',RandomForestClassifier(n_estimators=100))]
        stacking=StackingClassifier(estimators=estimatorss,final_estimator=final_est)
        return stacking

    def fit_combined(self,final_est):
        split=Split(self.data)
        X,y=split.split_to_x_y()
        X_scaled,y_rescaled=split.rescale(X,y)
        X_train,X_test,y_train,y_test=split.split(X_scaled,y_rescaled)
        
        combined=self.combine__classification(final_est)
        combined.fit(X_train,y_train)
        y_predicted=combined.predict(X_test)
        print("Accuracy of the model is ",accuracy_score(y_test,y_predicted))


    
        
        
       
    

    
       


    

       

    