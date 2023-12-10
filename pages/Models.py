from TrainTestSplit import Split
from TrainTestSplit import ModelAnalysis
from DataCleaning import DataCleaning
import streamlit as st
import matplotlib.pyplot as plt

dc=DataCleaning()
dc.read_csv()
dc.drop_columns()
dc.split()
dc.to_numeric('Systolic')
dc.to_numeric('Diastolic')
dc.label_encoding()
tts=Split(dc.data)
X,y=tts.split_to_x_y()
X_scaled,y_rescaled=tts.rescale(X,y)
X_train,X_test,y_train,y_test=tts.split(X_scaled,y_rescaled)

def main():
    st.title("Model Analysis")
    model_analysis=ModelAnalysis(X_train,y_train,X_test,y_test)
    result_df=model_analysis.summarize_models()
    #models result
    st.write("Model Summary:")
    st.write(result_df)
    #roc curve
    st.subheader("ROC curves")
    for model in result_df['Model']:
     st.write(f"###{model}")
     m_model=next((m[1] for m in model_analysis.models if m[0]==model),None)
     if m_model:
        y_pred_proba=m_model.predict_proba(X_test)
        model_analysis.roc_auc_plot(model,y_test,y_pred_proba)

if __name__ == "__main__":
    main()

