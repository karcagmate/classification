import streamlit as st
import sys
sys.path.append('projektmunka\DataCleaning.py')
from DataCleaning import DataCleaning
import matplotlib.pyplot as plt
st.set_page_config(page_title="Data preview",page_icon="📊")

st.title("Data preview")

st.sidebar.header("Data preview")
dc=DataCleaning()
#load original data
dc.read_csv()
st.subheader("Original Data")
st.write(dc.data)
#Data cleaning steps
st.subheader("Cleaned data")
if st.checkbox("Show cleaned data"):
   #cols_to_drop=st.multiselect("Select columns to drop", dc.data.columns)
   dc.drop_columns()
   dc.split()
   dc.to_numeric('Systolic')
   dc.to_numeric('Diastolic')
   dc.label_encoding()
   #st.subheader("Cleaned data")
   st.write(dc.data)

#Data analysis steps
st.subheader("Data Analysis")
#if st.checkbox("Columns Analysis"):
 #  st.subheader("Numeric and non numeric columns:")
  # st.write(dc.cols_analysis())
#if st.checkbox("Missing values"):
 # st.write( dc.missing_values())
#if st.checkbox("Search for duplicates"):
#   st.write(dc.searching_duplicates())

if st.checkbox("Detect outliers"):
   cols=["Age","Cholesterol","BMI","Systolic","Diastolic","Triglycerides"]
   cols_to_plot=st.selectbox("Select columns to plot",cols)
   fig,(ax1,ax2)=plt.subplots(1,2, figsize=(12,5))
   #plot=dc.data[cols_to_plot].plot(kind='box',figsize=(12,8))
   ax1.boxplot(dc.data[cols_to_plot])
   ax1.set_xlabel(cols_to_plot)
   ax1.set_ylabel('Values')
   #st.pyplot(fig)
   #fig,ax=plt.subplots()
   ax2.hist(dc.data[cols_to_plot], bins=20, edgecolor='black')
   ax2.set_xlabel(cols_to_plot)
   ax2.set_ylabel('Count')
   ax2.set_title('Distribution')
   st.pyplot(fig)
