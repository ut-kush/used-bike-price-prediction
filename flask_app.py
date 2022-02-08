#-------------------------------------------------------------------------------------------------------------------------------------------
#importing the necessary libraries
#-------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from flask import Flask, request, jsonify, render_template
from ml_model import encoder
import pickle

#-------------------------------------------------------------------------------------------------------------------------------------------
#importing the necessary libraries
#-------------------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
model = pickle.load(open('xgb.pkl', 'rb'))

#-------------------------------------------------------------------------------------------------------------------------------------------
#default page of the web-app
#-------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('home page.html')

#-------------------------------------------------------------------------------------------------------------------------------------------
#predict button
#-------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/result',methods=['POST'])
def predict():
    
    kms_driven=float(request.form.get('kms_driven'))
    owner=str(request.form.get('owner'))
    age=float(request.form.get('age'))
    power=float(request.form.get('power'))
    brand=str(request.form.get('brand'))
    
    owner_encoded,brand_encoded=encoder.transform([[owner,brand]])[0]
    
    prediction=model.predict([np.array([kms_driven,owner_encoded,age,power,brand_encoded])])
    output=round(np.exp(prediction[0]),2)
    
    return render_template('result.html',result='Price of bike is : {}'.format(output))
    
if __name__ == "__main__":
    app.run()
    app.debug=1
    
    