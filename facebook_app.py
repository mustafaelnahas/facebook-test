# importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# configure page
st.set_page_config(layout='wide') 


# loading data
df = pd.read_csv('facebook_ads.csv', encoding="ISO-8859-1")

# sidebar
option = st.sidebar.selectbox("Pick a choice:",['Home','EDA','ML'])

if option == 'Home':
    st.title("Facebook App")
    st.text('Author: @MustafaOthman')
    st.dataframe(df.head())

elif option == 'EDA':
    st.title("Facebook EDA")

    col1, col2 = st.columns(2)

    fig = px.scatter(data_frame=df, x='Time Spent on Site', y='Salary', color='Clicked')
    st.plotly_chart(fig)

    with col1:
        fig = px.violin(data_frame=df, x='Time Spent on Site')
        st.plotly_chart(fig)

    with col2:
        df['Clicked'].astype('O')
        fig = px.bar(data_frame=df, x=df['Clicked'])
        st.plotly_chart(fig)       

        # fig = plt.figure()
        # df['Clicked'].value_counts().plot(kind='bar')
        # st.pyplot(fig)

elif option == "ML":
    st.title("Ads Clicked Prediction")
    st.text("In this app, we will predict the ads click using salary and time spent on website")
    st.text("Please enter the following values:")

# building model
    time = st.number_input("Enter time spent on website")
    salary = st.number_input("Enter salary")
    btn = st.button("Submit")
    # df.drop(columns=['Names','emails','Country'], inplace=True)
    # X = df.drop("Clicked", axis=1)
    # y = df['Clicked']
    # ms = MinMaxScaler()
    # X = ms.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    ms = MinMaxScaler()
    clf = pickle.load(open('my_model.pkl','rb'))
    result = clf.predict(ms.fit_transform([[time, salary]]))

    if btn:
        if result == 1:
            st.write("Clicked")
        elif result == 0:
            st.write("Not Clicked")
