from TrainTestSplit import Split
from TrainTestSplit import ModelAnalysis
from DataCleaning import DataCleaning
import streamlit as st
import matplotlib.pyplot as plt
dc=DataCleaning()
@st.cache_data
def load_data():
 # data=dc.data
  dc.read_csv()
  dc.drop_columns()
  dc.split()
  dc.to_numeric('Systolic')
  dc.to_numeric('Diastolic')
  dc.label_encoding()
  return dc.data
tts=Split(load_data())
@st.cache_data

def split():
  X,y=tts.split_to_x_y()
  X_scaled,y_rescaled=tts.rescale(X,y)
  X_train,X_test,y_train,y_test=tts.split(X_scaled,y_rescaled)
  return X_train,y_train,X_test,y_test
X_train,y_train,X_test,y_test=split()
model_analysis=ModelAnalysis(X_train,y_train,X_test,y_test)

@st.cache_data
def load_result_df():
  result_df,models=model_analysis.summarize_models()
  return result_df,models
@st.cache_data
def load_roc_curves():
  #X_train,X_test,y_train,y_test=load_data()
  #model_analysis=ModelAnalysis(X_train,y_train,X_test,y_test)
  result_df,models=load_result_df()
  

  for model in result_df['Model']:
      st.write(f"{model}")
      m_model=next((m[1] for m in models if m[0]==model),None)
      if m_model:
        y_pred_proba=m_model.predict_proba(X_test)
        model_analysis.roc_auc_plot(model,y_test,y_pred_proba)

def main():
    
    st.title("Model Analysis")
     #result_df=load_result_df()
    #st.write(result_df)
    #models result
    if st.checkbox("Models Summary"):
     result_df,models=load_result_df()
     st.write(result_df)
     #st.write("Models Summary")
     #st.write(result_df)
    #roc curve
    st.subheader("ROC curves")
    if st.checkbox("Plot ROC curves"):
      load_roc_curves()

if __name__ == "__main__":
    main()

