import os
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog
import tkinter.font as font
from functools import partial
from pyresparser import ResumeParser
from sklearn import datasets, linear_model 
from flask import Flask, render_template
from flask import url_for, request, session, redirect
from flask_pymongo import PyMongo
from pymongo import MongoClient
import bcrypt
import spacy
import numpy as np
from functools import wraps
import pickle
import joblib

class train_model:
    
    def train(self):
        data =pd.read_csv('sampledata/training_dataset.csv')
        array = data.values

        for i in range(len(array)):
            if array[i][0]=="Male":
                array[i][0]=1
            else:
                array[i][0]=0


        df=pd.DataFrame(array)

        maindf =df[[0,1,2,3,4,5,6]]
        mainarray=maindf.values

        temp=df[7]
        train_y =temp.values
        
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
        self.mul_lr.fit(mainarray, train_y)
       
    def test(self, test_data):
        try:
            test_predict=list()
            for i in test_data:
                test_predict.append(int(i))
            y_pred = self.mul_lr.predict([test_predict])
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")
    def check_type(data):
        if type(data)==str or type(data)==str:
            return str(data).title()
        if type(data)==list or type(data)==tuple:
            str_list=""
            for i,item in enumerate(data):
                str_list+=item+", "
            return str_list
        else:   return str(data)

nlp = spacy.load("en_core_web_sm")
def prediction_result(aplcnt_name, cv_path, personality_values):
    applicant_data = {"Candidate Name":aplcnt_name,  "CV Location":cv_path} 
    age = personality_values[1]
    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)
    model = train_model()
    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")
    print(personality)
    data = ResumeParser(cv_path).get_extracted_data()
    try:
        del data['name']
        if len(data['mobile_number'])<10:
            del data['mobile_number']
    except:
        pass
    print("\n############# Resume Parsed Data #############\n")
    for key in data.keys():
        if data[key] is not None:
            print('{} : {}'.format(key,data[key]))
       
    


app = Flask(__name__)
# pickleFile=open("model.pkl","rb")
# regressor=pickle.load(pickleFile)
# model = pickle.load(open(model1.pkl))
# app.config['MONGO_DBNAME'] = 'carrerpredictorlogin'
app.config['MONGO_URI'] = 'mongodb+srv://S555600:sobha1809@careerpredictor.ltr6ht6.mongodb.net/?retryWrites=true&w=majority'

mongo = PyMongo(app)

# Database
client = MongoClient('mongodb://localhost:27017/')
db = client.careerpredictor
db.careerpredictordata


# Decorators
def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')  
  return wrap

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/login', methods=['POST','GET'])
def login():
    # users = mongo.db.users
    # login_user = users.find_one({'name' : request.form['username']})

    # if login_user:
    #     if bcrypt.hashpw(request.form['pass'].encode('utf-8'), login_user['password'].encode('utf-8')) == login_user['password'].encode('utf-8'):
    #         session['username'] = request.form['username']
    #         return redirect(url_for('dashboard'))

    # return 'Invalid username/password combination'
     return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/personalityprediction', methods=['POST','GET'])
def personalityprediction():
    
    a =  request.form.get('sName')
    # print(len(features))
    b =   request.form.get('cv')
    c = [request.form.get('openness'),request.form.get('neuroticism'),request.form.get('conscientiousness'),request.form.get('agreeableness'),request.form.get('extraversion')]
    if __name__ == "__main__":
        model = train_model()
        model.train()
        
    return render_template('personalityprediction.html')
# if __name__ == '__main__':
    
#     app.run()
@app.route('/result',methods = ['GET','POST'])
def result():
    if __name__ == "__main__":
        model =  personalityprediction()       
        # a = model.request.form.get('sName')
        # b= model.request.form.get('cv')
        # c = [model.request.form.get('openness'),model.request.form.get('neuroticism'),model.request.form.get('conscientiousness'),model.request.form.get('agreeableness'),model.request.form.get('extraversion')]
    return render_template('result.html')

@app.route('/jobprediction')
def jobprediction():
    return render_template('jobprediction.html')

@app.route('/salaryprediction')
def salaryprediction():
    return render_template('salaryprediction.html')

@app.route('/loginRedirect')
def loginRedirect():
    return render_template('loginRedirect.html')

if __name__ == '__main__':
    app.debug =  True
    app.run()
