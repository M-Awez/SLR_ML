import flask 
from flask import Flask
from flask import request,render_template
import numpy as np 
import pandas as pd
import pickle 

with open ("model.pkl",'rb') as f:
    m=pickle.load(f)

app=Flask(__name__)
@app.route('/')
def main_page():
    return render_template('index.html')
@app.route("/predict",methods=['GET','POST'])
def calculate_weight():
    
    a=float(request.form['height'])/100
    f=np.array([[a]])
    result=m.predict(f)[0][0]
    return render_template("index.html",result=round(float(result),2),a=a)
if __name__=="__main__":
    app.run(debug=True)