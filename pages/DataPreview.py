import streamlit as st
from DataCleaning import DataCleaning
import matplotlib.pyplot as plt
st.title("Data preview")
#st.set_page_config(page_title="Dataframe",page_icon="📊")
st.markdown("#Dataframe")
st.sidebar.header("Dataframe")
dc=DataCleaning()
#load original data
dc.read_csv()
st.subheader("Original Data")
st.write(dc.data)
#Data cleaning steps
st.subheader("Data Cleaning Steps")
if st.checkbox("Show cleaned data"):
   #cols_to_drop=st.multiselect("Select columns to drop", dc.data.columns)
   dc.drop_columns()
   dc.split()
   dc.to_numeric('Systolic')
   dc.to_numeric('Diastolic')
   dc.label_encoding()
   st.subheader("Cleaned data")
   st.write(dc.data)

#Data analysis steps
st.subheader("Data analysis steps")
if st.checkbox("Columns Analysis"):
   st.subheader("Numeric and non numeric columns:")
   st.write(dc.cols_analysis())
if st.checkbox("Missing values"):
  st.write( dc.missing_values())
if st.checkbox("Search for duplicates"):
   st.write(dc.searching_duplicates())

if st.checkbox("Detect outliers"):
   cols=["Age","Cholesterol","BMI","Systolic","Diastolic","Triglycerides"]
   cols_to_plot=st.selectbox("Select columns to plot",cols)
   fig,ax=plt.subplots()
   #plot=dc.data[cols_to_plot].plot(kind='box',figsize=(12,8))
   ax.boxplot(dc.data[cols_to_plot])
   ax.set_xlabel(cols_to_plot)
   ax.set_ylabel('Values')
   st.pyplot(fig)
   fig,ax=plt.subplots()
   ax.hist(dc.data[cols_to_plot], bins=20, edgecolor='black')
   ax.set_xlabel(cols_to_plot)
   ax.set_ylabel('Count')
   ax.set_title('Distribution')
   st.pyplot(fig)
