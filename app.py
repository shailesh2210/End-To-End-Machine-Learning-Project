
from flask import Flask, request , app , json , render_template , jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model
model = pickle.load(open("regmodel.pkl","rb"))
scalar = pickle.load(open("pickle.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = [float(x) for x in request.form.values()]
#     input = scalar.transform(np.array(data).reshape[1,-1])
#     print(input)
#     output = model.predict(input)[0]
#     return render_template("home.html", prediction_text = " The house price is {} ".format(output))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output= model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)


