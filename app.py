import pandas as pd
from tkinter import Tk
from pyresparser import ResumeParser
from sklearn import linear_model 
from flask import request, session, redirect, Flask, render_template, flash
from pymongo import MongoClient
import spacy
from functools import wraps
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
class train_model:  
    # def __init__(self):
    #     self.mul_lr = None

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
        trainy = temp
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter =1000)
        self.mul_lr.fit(mainarray, trainy)
       
    def test(self, test_data):
        try:
            nlp = spacy.load("en_core_web_sm")
            test_predict=list()
            for i in test_data:
                test_predict.append(int(i))
            print(test_predict)
            # w = 
            # print(w)
            y_pred = self.mul_lr.predict([test_predict])
            print(y_pred)
            return y_pred
        except:
            print("All Factors For Finding Personality Not Entered!")
    def check_type(self,data):
        if type(data)==str or type(data)==str:
            return str(data).title()
        if type(data)==list or type(data)==tuple:
            str_list=""
            for i,item in enumerate(data):
                str_list+=item+", "
            return str_list
        else:   
            return str(data)

def prediction_result(aplcnt_name, cv_path, personality_values):
    applicant_data = {"Candidate Name":aplcnt_name,  "CV Location":cv_path} 
    age = personality_values[1]
    print(age)
    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)
    model = train_model()
    personality = model.test(personality_values)
    print("\n############# Predicted Personality #############\n")
    print(personality)
    data = ResumeParser(cv_path).get_extracted_data()
    try:
        if data is not None:
            del data['name']
            if len(data['mobile_number'])<10: # type: ignore
                del data['mobile_number']
    except:
        pass
    print("\n############# Resume Parsed Data #############\n")
    for key in data.keys():
        if data[key] is not None:
            print('{} : {}'.format(key,data[key]))

app = Flask(__name__)
app.config[UPLOAD_FOLDER] = 'uploads'
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    app.debug =  True

client = MongoClient('mongodb://localhost:27017/')
db = client['careerpredictor']
users_collection = db['users']
employee_collection = db['organization']
# users_collection.insert_one({'username': 'S555600@nwmissouri.edu', 'password': '123'})
users_collection.create_index("username", unique=True)
# employee_collection.insert_one({'organization': 'sample','username':'S555600@nwmissouri.edu','password': '123'})
employee_collection.create_index("username", unique=True)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/loginRedirect')
def loginRedirect():
    return render_template('loginRedirect.html')

@app.route('/registerRedirect')
def registerRedirect():
    return render_template('registerRedirect.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print('yessss')
        username = request.form['email']
        password = request.form['password']
        print(username)
        print(password)
        # Query the MongoDB collection for the username and password
        user = users_collection.find_one({'username': username})
        print(user)
        if user is not None:
            if user['username']==username and user['password'] == password:
                return redirect('dashboard')
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/loginEmployee', methods=['GET', 'POST'])
def loginEmployee():
    if request.method == 'POST':
        organization = request.form['organization']
        username = request.form['username']
        password = request.form['password']

        # Query the MongoDB collection for the username and password
        user = employee_collection.find_one({organization :'organization' ,'username': username, 'password': password})

        if user:
            # Successful login
            return redirect('dashboard')
        else:
            # Invalid credentials
            return render_template('loginEmployee.html', error='Invalid username or password')

    return render_template('loginEmployee.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['email']
        print(username)
        password = request.form['password']
        print(password)
        password1 = request.form['password-reenter']
        print(password1)
        if password != password1 :
            error_message = 'Passwords do not match.'
            print(error_message)
            return render_template('register.html', error='Passwords do not match')
                # Insert the new user into the MongoDB collection
        else:
            users_collection.insert_one({'username': username, 'password': password1})
            return redirect('/login')
    return render_template('register.html')

@app.route('/employeeRegister', methods=['GET', 'POST'])
def employeeRegister():
    if request.method == 'POST':
        organization = request.form['organization']
        username = request.form['email']
        password = request.form['password']
        password1 = request.form['password-reenter']
        if password == password1 :
            if db['organization'].find_one({'email': username}):
                return render_template('register.html', error='Username already exists')
                # Insert the new user into the MongoDB collection
            else:
                db['organization'].insert_one({'username': username, 'password': password1})
                return redirect('/login')
        return render_template('employeeRegister.html',error='passwords should match')
    return render_template('employeeRegister.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/personalityprediction', methods=['POST','GET','PUT'])
def personalityprediction():  
    if request.method == 'POST':
        print('YESSSSS')
        a =  request.form.get('sName')
        # print(a)
        q = request.form.get('cv')
        b =   request.files['cv']
        # b1.save('uploads/' + b1.filename)
        # b.save('uploads/' + b.filename)
        g = request.form.get('gender')
        print(g)
        # print(b)
        c = [request.form.get('gender'),request.form.get('age'),request.form.get('openness'),request.form.get('neuroticism'),request.form.get('conscientiousness'),request.form.get('agreeableness'),request.form.get('extraversion')]
        # print(c)
        model = train_model()
        model.train()      
        return render_template('result.html',name=a,cv=b,list=c)
    return render_template('personalityprediction.html')

@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        # model = personalityprediction()   
        a1 = request.form.get('sName')
        print(a1)
        print('in Here ')
        if request.form.get('gender')=='female':
            r = 0
        else:
            r=1
        ag = request.form.get('age')
        q1 = request.form.get('openness')
        q2 = request.form.get('neuroticism')
        q3 = request.form.get('conscientiousness')
        q4 = request.form.get('agreeableness')
        q5 = request.form.get('extraversion')
        c1 = [r,ag,q1,q2,q3,q4,q5 ]
        print(c1)
        b1 = request.files.get('cv')
        d = request.form.get('cv')
        path = "uploads/" + d # type: ignore
        b1.save(path)
        print('came here')     
        s = prediction_result(a1,b1,c1)
        return render_template('result.html')
    return render_template('result.html')

