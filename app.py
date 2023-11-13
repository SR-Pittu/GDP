import io
from blinker import receiver_connected
import pandas as pd
import os
import pickle
from tkinter import Tk
from cryptography.fernet import Fernet
from pymongo.errors import DuplicateKeyError
import subprocess
from pyresparser import ResumeParser
from resume_parser import resumeparse
from sklearn import linear_model
from flask import request, session, redirect, Flask, render_template, flash
from pymongo import MongoClient
from sklearn.discriminant_analysis import StandardScaler
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
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import spacy

class train_model:
    def __init__(self):
        self.mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)

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
        mainarray = maindf
        temp = df[7]
        train_y = temp
        self.mul_lr.fit(mainarray.values, train_y)
        # arr.reshape(-1,1)
        # print(self.mul_lr.predict([arr]))


    def test(self, test_data):
        model = train_model()
# Train the logistic regression model
        nlp = spacy.load("en_core_web_sm")
        test_predict = list()
        for i in test_data:
            test_predict.append(int(i))
        print(test_predict)
        # print(self.mul_lr.predict([test_predict]))
        y_pred=self.mul_lr.predict([test_predict])
        # print(y_pred)
        return y_pred


def prediction_result(aplcnt_name, cv_path, personality_values):
    applicant_data = {"Candidate Name":aplcnt_name,  "CV Location":cv_path} 
    print("\n############# Candidate Entered Data #############\n")
    print(applicant_data, personality_values)
    model = train_model()
    model.train()
    personality = model.test(personality_values)
    print(personality)
    print("\n############# Predicted Personality #############\n")
    print(personality)
    # fp = io.open(cv_path,'r')   
    print(cv_path) 
    path = 'uploads/' + cv_path
    data = ResumeParser(path).get_extracted_data()
    str = ""
    try:
        del data['name']
        if len(data['mobile_number'])<10: # type: ignore
            del data['mobile_number'] 
    except:
        pass
    print("\n############# Resume Parsed Data #############\n")
    for key in data.keys():
        if data[key] is not None:
            str += "\n" + ('{} : {}'.format(key,data[key])) # type: ignore
    return (str,personality)

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


app = Flask(__name__)
app.secret_key = '1809'

if __name__ == '__main__':
    # Initialize the model outside of route functions
    app.secret_key = '1809'
    model = train_model()
    app.run(host='0.0.0.0', debug=True)
    app.debug =  True

client = MongoClient('mongodb://localhost:27017/')
db = client['careerpredictor']
users_collection = db['users']
employee_collection = db['organization']
# users_collection.insert_one({'username': 'S555600@nwmissouri.edu', 'password': '123'})
# users_collection.create_index("username", unique=True)
# employee_collection.insert_one({'organization': 'sample','email':'S555600@nwmissouri.edu','password': '123'})
employee_collection.create_index("username", unique=True)

def is_user_logged_in():
    return 'logged_in' in session
    
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
                print("session")
                session['user_id'] = username
                print(session['user_id'])
                return redirect('dashboard')
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/loginEmployee', methods=['GET', 'POST'])
def loginEmployee():
    if request.method == 'POST':
        print("Sucess")
        organization = request.form['organization']
        print(organization)
        username = request.form['email']
        print(username)
        password = request.form['password']
        print(password)

        # Query the MongoDB collection for the username and password
        user = employee_collection.find_one({'organization' :organization ,'email': username, 'password': password})

        if user:
            # Successful login
            print("Done")
            return redirect('jobpage')
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
        else:
            users_collection.insert_one({'username': username, 'password': password1})
            return redirect('/login')
    return render_template('register.html')

@app.route('/employeeRegister', methods=['GET', 'POST'])
def employeeRegister():
    if request.method == 'POST':
        print('entered here')
        organization = request.form['organization']
        print(organization)
        username = request.form['email']
        print(username)
        password = request.form['password']
        print(password)
        password1 = request.form['password-reenter']
        print(password1)
        if password == password1 :
            if employee_collection.find_one({'email': username}):
                return render_template('employeeRegister.html', error='Username already exists')
            else:
                employee_collection.insert_one({'organization': organization,'email': username, 'password': password1})
                return redirect('/loginEmployee')
        return render_template('employeeRegister.html',error='passwords should match')
    return render_template('employeeRegister.html')

@app.route('/dashboard')
def dashboard():
    if is_user_logged_in():
        return render_template('dashboard.html')
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.clear()
    return render_template('home.html')

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
        return render_template('result.html',name=a,cv=b,list=c)
    return render_template('personalityprediction.html')

@app.route('/result',methods = ['POST','GET','PUT'])
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
        b1 = request.files['cv']
        file_path = os.path.join('uploads/', b1.filename) # type: ignore
        filename = os.path.basename(file_path)
        b1.save(file_path)
        print('came here') 
        model = train_model()   
        model.train()
        s = prediction_result(a1,filename,c1)
        print(s)
        user = session.get('user_id')
        print(user)
        persona =  np.array(s[1])
        p = persona[0]
        # users_collection.update({"username": user}, { $set : {"personality" : 1 } } )
        users_collection.update_one({"username": user}, {"$set": {"Name": a1 ,"personality": p,"Resume": file_path}})
        return render_template('result.html',a1=a1,ag = ag,s=s )
    return render_template('result.html')

@app.route('/jobprediction', methods=['POST','GET','PUT'])
def jobprediction():
    if request.method =='POST':
        list =[]
        a1 = request.form.get("LogicalSkills")
        list.append(int(a1)) # type: ignore
        a2 = request.form.get("CodingSkills")
        list.append(int(a2)) # type: ignore
        a3 = request.form.get("Hackathons")
        list.append(int(a3)) # type: ignore
        a4 = request.form.get("PublicSpeaking")
        list.append(int(a4)) # type: ignore
        a5 = request.form.get('j')
        list.append(checkValue(a5))
        a6 = request.form.get('j1')
        list.append(checkValue(a6))
        a7 = request.form.get('j2')
        list.append(checkValue(a7))
        a8 = request.form.get('j3')
        list.append(checkValue(a8))
        a9 = request.form.get('j4')
        list.append(checkValue(a9))
        a10 = request.form.get('j5')
        list.append(valuesCheck(a10))
        a11 = request.form.get('j6')
        list.append(valuesCheck(a11))
        a12 = request.form.get('j7')
        if(a12=="Hard"):
            list.append(1)
            list.append(0)
        else:
            list.append(0)
            list.append(1)
        a13 = request.form.get('j8')
        if(a13=="Management"):
            list.append(1)
            list.append(0)
        else:
            list.append(0)
            list.append(1)
        a14 = request.form.get('j9')
        list.append(a14)
        a15 = request.form.get('j10')
        list.append(a15)
        a16 = request.form.get('j11') 
        list.append(a16)
        a17 = request.form.get('j12')
        list.append(a17)
        a18 = request.form.get('j13')
        list.append(a18)
        a19 = request.form.get('j14')
        list.append(a19)
        print(list)
        string_to_code = {}
        code_counter = 0
        for item in list:
            if isinstance(item, str):
                if item not in string_to_code:
                    string_to_code[item] = code_counter
                    code_counter += 1

        # Create a new list with the categorical codes
        categorical_codes = [string_to_code.get(item, item) for item in list]

        print(categorical_codes)
        name = request.form.get("name")
        print(name)
        b = jobPredict()
        b.train_decision_tree()
        a = b.predict_decision_tree(categorical_codes)
        user = session.get('user_id')
        print(user)
        persona =  np.array(a)
        p = persona[0]
        # users_collection.update({"username": user}, { $set : {"personality" : 1 } } )
        users_collection.update_one({"username": user}, {"$set": {"Name": name ,"JobTitle": p}})
        return render_template("jobtitleresult.html",a=a, name= name)
    return render_template("jobprediction.html")

@app.route('/jobtitleresult',methods = ['POST'])
def jobtitleresult():
    if request.method == 'POST':
        return render_template('jobtitleresult.html')
    return render_template("jobtitleresult.html")

@app.route('/jobpage',methods = ['GET','POST'])
def jobpage():
    print("Hello")
    if request.method == 'GET':
        print("YESSS")
        persons = users_collection.find()
        print(persons)
        return render_template('jobpage.html',persons=persons)
    return render_template("jobpage.html")


@app.route('/salaryprediction',methods = ['POST','GET','PUT'])
def salaryprediction():
    if request.method == 'POST':
        a = request.form.get("country") 
        print(a)
        b = request.form.get("education") 
        print(b)
        c = request.form.get("experience") 
        print(c)
        data = load_model()
        regressor = data["model"]
        le_country = data["le_country"]
        le_education = data["le_education"]
        X = np.array([[a, b, c ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        salary = regressor.predict(X)
        name = request.form.get('name')
        print(name)
        print(f"The estimated salary is ${salary[0]:.2f}") 
        user = session.get('user_id')
        print(user)
        persona =  np.array(salary)
        p = persona[0]
        # users_collection.update({"username": user}, { $set : {"personality" : 1 } } )
        users_collection.update_one({"username": user}, {"$set": {"Name": name ,"SalaryPredicted": p}})

        return render_template('salaryprediction.html',salary=salary)
    return render_template('salaryprediction.html')

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data



def checkValue(c):
    if(c=="Yes"):
        return 1
    else:
        return 0
def valuesCheck(b):
    if(b=="Poor"):
        return 0
    elif(b=="Medium"):
        return 1
    else:
        return 2


def extract_skills_from_cv(cv_text):
    # Initialize an empty list for resume tokens
    resume_tokens = []

    # Tokenize the resume text
    for token in nlp(cv_text):
        if not token.is_stop and not token.is_punct:
            resume_tokens.append(token.text.lower())

    return resume_tokens
class jobPredict:
    def __init__(self):
        self.dtree = DecisionTreeClassifier(random_state=1)
        self.xgb = XGBClassifier(random_state=42, learning_rate=0.02, n_estimators=300)
        self.df = pd.read_csv('sampledata/mldata.csv')
        
    def train_decision_tree(self):
        df = pd.read_csv('sampledata/mldata.csv')
        df.head()
        cols = df[["self-learning capability?", "Extra-courses did","Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
        for i in cols:
            cleanup_nums = {i: {"yes": 1, "no": 0}}
            df = df.replace(cleanup_nums) 
        print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())
        mycol = df[["reading and writing skills", "memory capability score"]]
        for i in mycol:
            cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
            df = df.replace(cleanup_nums)

        category_cols = df[['certifications', 'workshops', 'Interested subjects', 'interested career area ', 'Type of company want to settle in?', 
                            'Interested Type of Books']]
        for i in category_cols:
            df[i] = df[i].astype('category')
            df[i + "_code"] = df[i].cat.codes # type: ignore

        print("\n\nList of Categorical features: \n" , df.select_dtypes(include=['object']).columns.tolist())
        df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])
        df.head()
    
        feed = df[['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?','Extra-courses did', 
                'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',  
                'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested subjects_code', 'Interested Type of Books_code', 'certifications_code', 
                'workshops_code', 'Type of company want to settle in?_code',  'interested career area _code', 'Suggested Job Role']]

        # Taking all independent variable columns
        df_train_x = feed.drop('Suggested Job Role',axis = 1)

        # Target variable column
        df_train_y = feed['Suggested Job Role']
    
        x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)

        # dtree = DecisionTreeClassifier(random_state=1)
        dtree = self.dtree.fit(x_train, y_train)

        y_pred = dtree.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        print("confusion matrics=",cm)
        print("accuracy=",accuracy*10)
        userdata = [['7','6','6','8','3','5','4', '4', '7', '3', '3', '6','8','7','5','7','4','5','6','8','8']]
        ynewclass = dtree.predict(userdata)
        ynew = dtree.predict_proba(userdata)
        print(ynewclass)
        print("Probabilities of all classes: ", ynew)
        print("Probability of Predicted class : ", np.max(ynew))

    def train_xgboost(self):
        feed = self.df[['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?','Extra-courses did', 
                'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',  
                'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested subjects_code', 'Interested Type of Books_code', 'certifications_code', 
                'workshops_code', 'Type of company want to settle in?_code',  'interested career area _code', 'Suggested Job Role']]

        df_train_x = feed.drop('Suggested Job Role',axis = 1)

        # Target variable column
        df_train_y = feed['Suggested Job Role']
    # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.2, random_state=42)

        # Encode the string labels to numeric labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        df_train_x = feed.drop('Suggested Job Role',axis = 1)
        df_train_y = feed['Suggested Job Role']
        xgb = XGBClassifier(random_state = 42, learning_rate=0.02, n_estimators=300)
        xgb.fit(X_train.values, y_train_encoded)
        xgb_y_pred = xgb.predict(X_test)
        xgb_cm = confusion_matrix(y_test_encoded, xgb_y_pred)
        xgb_accuracy = accuracy_score(y_test_encoded, xgb_y_pred)
        print("confusion matrics=",xgb_cm)
        print(xgb_y_pred)
        print("accuracy=",xgb_accuracy*10)

    def svcTrain(self):
        feed = self.df[['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?','Extra-courses did', 
                'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',  
                'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested subjects_code', 'Interested Type of Books_code', 'certifications_code', 
                'workshops_code', 'Type of company want to settle in?_code',  'interested career area _code', 'Suggested Job Role']]

        df_train_x = feed.drop('Suggested Job Role',axis = 1)

        # Target variable column
        df_train_y = feed['Suggested Job Role']
    # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.2, random_state=42)

        svm_classifier = svm.SVC()
        svm_classifier.fit(x_train, y_train)
        y_pred = svm_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy*10)

    def rfTrain(self):
        feed = self.df[['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?','Extra-courses did', 
                'Taken inputs from seniors or elders', 'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',  
                'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested subjects_code', 'Interested Type of Books_code', 'certifications_code', 
                'workshops_code', 'Type of company want to settle in?_code',  'interested career area _code', 'Suggested Job Role']]

        df_train_x = feed.drop('Suggested Job Role',axis = 1)

        # Target variable column
        df_train_y = feed['Suggested Job Role']
    # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.2, random_state=42)


        rf = RandomForestClassifier(random_state = 10)
        rf.fit(x_train, y_train)
        rfc_y_pred = rf.predict(x_test)
        rfc_cm = confusion_matrix(y_test,rfc_y_pred)
        rfc_accuracy = accuracy_score(y_test,rfc_y_pred)
        print("confusion matrics=",rfc_cm)
        print("  ")
        print("accuracy=",rfc_accuracy*10)

    def predict_decision_tree(self, data):
        # Use the self.dtree model to predict job titles for input data
        for i in data:
            i = str(i)
        print(data)
        value = self.dtree.predict([data])        
        return value
    
