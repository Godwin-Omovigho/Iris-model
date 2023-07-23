import streamlit as st
import pickle
import numpy as np


def load_data():
    data=pickle.load(open('iris_model.pkl','rb'))

    return data


data=load_data()

def show_predict_page():
    st.title("Iris Species Classification")

    st.write("""### We need some information to Iris Species""")

    SepalLength=st.number_input("Sepal Length",4.0,8.0,4.0) 
    SepalWidth=st.number_input("Sepal Width",2.0,4.5,2.0 )
    PetalLength=st.number_input("Petal Length",1.0,7.0,1.0)
    PetalWidth=st.number_input('Petal Width',0.1,2.5,0.1)

    ok=st.button('Show Iris Specie')

    X=np.array([[SepalLength,SepalWidth,PetalLength,PetalWidth]])
    X = X.astype(float)

    if ok:
        
        model=data.predict(X)

        if model==0:
            return st.subheader("The Iris Specie is Iris-setosa")
        if model==1:
            return st.subheader("The Iris Specie is Iris-versicolor")
        if model==2:
            return st.subheader("The Iris Specie is Iris-virginica")
        
        