import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

model = RandomForestClassifier()

# Fit the model
model.fit(X, Y)

# Create the Streamlit app
st.title("Iris Flower Type Prediction App")
st.header("Enter the following parameters to predict the type of iris flower:")

# Add input fields for the features
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Define the prediction button
if st.button("Predict"):
    # Make a prediction
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_iris = iris.target_names[prediction[0]]
    
    # Display the predicted iris type
    st.write(f"The type of iris flower is {predicted_iris}")
