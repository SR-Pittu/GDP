import pandas as pd
from tkinter import Tk
from pyresparser import ResumeParser
from resume_parser import resumeparse
from sklearn import linear_model 
from flask import request, session, redirect, Flask, render_template, flash
from pymongo import MongoClient
import spacy
from functools import wraps
from fileinput import filename
import os
from werkzeug.utils import secure_filename
import PyPDF2

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()

    return text

def extract_job_title(resume_path):
    nlp = spacy.load("en_core_web_sm")
    text = extract_text_from_pdf(resume_path)

    # Process the text using spaCy
    doc = nlp(text)

    # Define a list of common job title keywords
    job_title_keywords = ["data scientist", "software engineer", "business analyst", "product manager"]

    # Extract job titles
    job_titles = []
    for token in doc:
        if any(keyword in token.text.lower() for keyword in job_title_keywords):
            job_titles.append(token.text)

    # If multiple job titles are found, you can choose the first one
    if job_titles:
        return job_titles[0]
    else:
        return "Job title not found"

class train_model: 
    def __init__(self):
        self.mul_lr = linear_model.LogisticRegression()
    def train(self):
        data = pd.read_csv('sampledata/training_dataset.csv')
        array = data.values
        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0
        df = pd.DataFrame(array)
        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values
        temp = df[7]
        trainy = temp
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainarray, trainy)

    def test(self, test_data):
        nlp = spacy.load("en_core_web_sm")
        test_predict = list()
        for i in test_data:
            test_predict.append(int(i))
        print(test_predict)
        y_pred = self.mul_lr.predict([test_predict])
        print(y_pred)
        return y_pred
# In this modified code, the mul_lr attribute is initialized in the __init__ constructor of the train_model class. This should resolve the attribute error you were encountering. Make sure to update your code accordingly.






        # except:
        #     print("All Factors For Finding Personality Not Entered!")
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
    # Initialize the model outside of route functions
    model = train_model()
    model.train()
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

@app.route('/result',methods = ['POST','GET','PUT'])
def result():
    if request.method == 'POST':
        # model = personalityprediction()   
        a1 = request.form.get('sName')
        print(a1)
        print('in Here ')
        model = train_model()
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
        b1 = request.files['cv']
        b1.save(b1.filename) # type: ignore
        print('came here')    
        # model.train() 
        s = prediction_result(a1,b1,c1)
        print(s)
        return render_template('result.html')
    return render_template('result.html')

@app.route('/jobprediction', methods=['POST','GET','PUT'])
def jobprediction():
    if request.method == 'POST':   
        # print("Hello")
        name = request.form.get('name')     
        print(name)
        b = request.files['cv']
        print(b)
        b.save(b.filename) # type: ignore
        model = jobtrain()
        model.train_job()
        s = model.predict_job(b)
        print(s)
        return render_template("jobtitleresult.html",name=name,s=s)
    return render_template("jobprediction.html")

@app.route('/jobtitleresult',methods = ['POST'])
def jobtitleresult():
    return render_template("jobtitleresult.html")

@app.route('/salaryprediction',methods = ['POST','GET'])
def salaryprediction():
    if request.method == 'POST':
        resume_path = "path_to_your_resume.pdf"
        job_title = extract_job_title(resume_path)
        print("Extracted Job Title:",job_title)
        return render_template('salaryprediction.html')
    return render_template('salaryprediction.html')

class jobtrain:
    def train_job(self):
        data = pd.read_csv('sampledata/jobtitle.csv')
        ar = data.values
        df = pd.DataFrame(ar)
        tem = df[0]
        ty = tem
        maindf = df[[1]]
        mainar = maindf.values
        self.mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainar,ty)
    
    def test_job(self,test_data):
        test_predict = list()
        print(test_predict)
        y_pred = self.mul_lr.predict([test_predict])
        print(y_pred)
        return y_pred
    
    def predict_job(self,cv_path):
        data = ResumeParser(cv_path).get_extracted_data()
        str = ""
        for key in data.keys():
            if data[key] is not None:
                str+=key
                print('{} : {}'.format(key,data[key]))
        
        model = jobtrain()
        model.train_job()
        jtp = model.test_job(str)