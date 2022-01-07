
from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialise the Flask app
app = Flask(__name__)

# Use pickle to load in the pre-trained model
filename = "model_7_xgb.pkl"
model = pickle.load(open(filename, "rb"))

# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():

    if request.method == "POST":
        # Extract the input from the form
        age = request.form.get("age")
        cea = request.form.get("cea")
        ibil = request.form.get("ibil")
        neu = request.form.get("neu")
        meno = request.form.get("meno")
        ca125 = request.form.get("ca125")
        alb = request.form.get("alb")
        
        # Create DataFrame based on input
        input_variables = pd.DataFrame([[age, cea, ibil,neu,meno,ca125,alb]],
                                       columns=['Age','CEA','IBIL','NEU','Menopause','CA125','ALB'],
                                       dtype=float,
                                       index=['input'])
        
        # Get the model's prediction
        # Given that the prediction is stored in an array we simply extract by indexing
        prediction = model.predict(input_variables)
    
        # We now pass on the input from the from and the prediction to the index page
        return render_template("index.html", original_input={'Age':age,'CEA':cea,'IBIL':ibil,'NEU':neu,'Menopause':meno,'CA125':ca125,'ALB':alb}, result=True, prediction=prediction)
    # If the request method is GET
    return render_template("index.html", result=False)
