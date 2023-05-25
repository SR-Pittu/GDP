from flask import Flask, render_template
from flask import url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt
app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'CareerPredictor'
app.config['MONGO_URI'] = 'mongodb+srv://S555600:sobha1809@careerpredictor.ltr6ht6.mongodb.net/?retryWrites=true&w=majority'

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.users
    login_user = users.find_one({'name' : request.form['username']})

    if login_user:
        if bcrypt.hashpw(request.form['pass'].encode('utf-8'), login_user['password'].encode('utf-8')) == login_user['password'].encode('utf-8'):
            session['username'] = request.form['username']
            return redirect(url_for('index'))

    return 'Invalid username/password combination'


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'name' : request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name' : request.form['username'], 'password' : hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('index'))
        
        return 'That username already exists!'

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/personalityprediction')
def personalityprediction():
    return render_template('personalityprediction.html')

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