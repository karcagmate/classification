import streamlit as st
#from DataCleaning import DataCleaning

st.set_page_config(
    page_title="Hello",
    
)
st.write("Welcome!")
#st.sidebar.success("select a demo.")


if st.button("Links"):
    link="https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset"
    st.write(f"Dataset:\n {link}")
    glink="https://github.com/karcagmate/classification"
    st.write(f"\n\nGithub Repo:\n{glink}\n")
    