import pandas as pd
from tkinter import *
from pyresparser import ResumeParser
from sklearn import linear_model 
from flask import request, session, redirect, Flask, render_template
from pymongo import MongoClient
import spacy
from functools import wraps

app = Flask(__name__)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    app.run(host='localhost', port=9874)
    app.debug =  True

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
       
    
client = MongoClient('mongodb://localhost:27017/')
db = client['careerpredictor']
users_collection = db['users']
employee_collection = db['organization']
# users_collection.insert_one({'username': 'S555600@nwmissouri.edu', 'password': '123'})
# users_collection.create_index("username", unique=True)
# employee_collection.insert_one({'organization': 'sample','username':'S555600@nwmissouri.edu','password': '123'})
# employee_collection.create_index("username", unique=True)



# # Decorators
# def login_required(f):
#   @wraps(f)
#   def wrap(*args, **kwargs):
#     if 'logged_in' in session:
#       return f(*args, **kwargs)
#     else:
#       return redirect('/')  
#   return wrap

@app.route('/')
def home():
    return render_template('home.html')
    
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
    a =  request.form.get('sName')
    # print(len(features))
    b =   request.form.get('cv')
    c = [request.form.get('openness'),request.form.get('neuroticism'),request.form.get('conscientiousness'),request.form.get('agreeableness'),request.form.get('extraversion')]
    model = train_model()
    model.train()      
    return render_template('personalityprediction.html',name=a,cv=b,list=c)
# # if __name__ == '__main__':
    
#     app.run()
@app.route('/result',methods = ['POST'])
def result():
    model =  personalityprediction()       
        # a = model.request.form.get('sName')
        # b= model.request.form.get('cv')
        # c = [model.request.form.get('openness'),model.request.form.get('neuroticism'),model.request.form.get('conscientiousness'),model.request.form.get('agreeableness'),model.request.form.get('extraversion')]
    return render_template('result.html')

app = Flask(__name__)

@app.route('/jobprediction', methods=['POST','GET','PUT'])
def jobprediction():
    # if request.method == 'POST':   
    #     a=   request.form.get('name')     
    #     print(a)
    return render_template("jobprediction.html")


@app.route('/salaryprediction')
def salaryprediction():
    return render_template('salaryprediction.html')

@app.route('/loginRedirect')
def loginRedirect():
    return render_template('loginRedirect.html')

@app.route('/registerRedirect')
def registerRedirect():
    return render_template('registerRedirect.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    app.run(host='localhost', port=9874)
    app.debug =  True
    