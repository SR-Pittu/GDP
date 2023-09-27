import pandas as pd
from tkinter import Tk
import subprocess
from pyresparser import ResumeParser
from resume_parser import resumeparse
from sklearn import linear_model 
from flask import request, session, redirect, Flask, render_template, flash
from pymongo import MongoClient
import spacy
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    if request.method =='POST':
        return render_template("jobtitleresult.html",a = predict_job_title_from_cv())
    return render_template("jobprediction.html")

@app.route('/jobtitleresult',methods = ['POST'])
def jobtitleresult():
    if request.method == 'POST':
        return render_template('jobtitleresult.html')
    return render_template("jobtitleresult.html")

@app.route('/salaryprediction',methods = ['POST','GET'])
def salaryprediction():
    if request.method == 'POST':
        resume_path = "path_to_your_resume.pdf"
        # job_title = extract_job_title(resume_path)
        # print("Extracted Job Title:",job_title)
        return render_template('salaryprediction.html')
    return render_template('salaryprediction.html')


def extract_skills_from_cv(cv_text):
    # Initialize an empty list for resume tokens
    resume_tokens = []

    # Tokenize the resume text
    for token in nlp(cv_text):
        if not token.is_stop and not token.is_punct:
            resume_tokens.append(token.text.lower())

    return resume_tokens

def predict_job_title_from_cv():
    df = pd.read_csv('sampledata/mldata.csv')
    df.head()
    cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
    for i in cols:
        cleanup_nums = {i: {"yes": 1, "no": 0}}
        df = df.replace(cleanup_nums)
    

    mycol = df[["reading and writing skills", "memory capability score"]]
    for i in mycol:
        cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
        df = df.replace(cleanup_nums)


    # Label Encoding
    category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                        'Interested Type of Books']]
    for i in category_cols:
        df[i] = df[i].astype('category')
        df[i + "_code"] = df[i].cat.codes # type: ignore
    feed = df[['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?','Extra-courses did', 
           'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',  
            'Interested subjects_code', 'Interested Type of Books_code', 'certifications_code', 
           'workshops_code', 'Type of company want to settle in?_code',  'interested career area _code',
             'Suggested Job Role']]
    
    # Taking all independent variable columns
    df_train_x = feed.drop('Suggested Job Role',axis = 1)

    # Target variable column
    df_train_y = feed['Suggested Job Role']

    # Train-Test Splitting
    x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)

    #Random Forest Classifier
    clf3 = RandomForestClassifier(n_estimators=100) 
    clf3 = clf3.fit(x_train, y_train)
   
    categorical_col = df[['self-learning capability?', 'Extra-courses did','reading and writing skills', 'memory capability score', 
                      'Taken inputs from seniors or elders', 'Management or Technical', 'hard/smart worker', 'worked in teams ever?', 
                      'Introvert', 'interested career area ']]
    for i in categorical_col:
        print(df[i].value_counts(), end="\n\n")

# Taking all independent variable columns
    df_train_x = feed.drop('Suggested Job Role',axis = 1)

    # Target variable column
    df_train_y = feed['Suggested Job Role']

    x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)

    dtree = DecisionTreeClassifier(random_state=1)
    dtree = dtree.fit(x_train, y_train)

    y_pred = dtree.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    print("confusion matrics=",cm)
    print("  ")
    print("accuracy=",accuracy*10)
    userdata = [['7','6','6','8','3','5','4', '4', '7', '3', '3', '6','8', 
                    ,'7','4','5','6','8','8']]
    ynewclass = dtree.predict(userdata)
    ynew = dtree.predict_proba(userdata)
    print(ynewclass)
    return ynewclass
    