#-----------------------------------------------extra_modules-----------------------------------------------------------
import os

#-------------------------------------------------model_code------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn import model_selection, metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score , classification_report, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import *
from sklearn import metrics

csv_filename = "pollution_us_2000_2016.csv"

df = pd.read_csv(csv_filename)
df = df.drop(['Unnamed: 0','Address','Site Num','State','County','City','Date Local','NO2 Units','O3 Units','SO2 Units','CO Units','NO2 Mean'
              ,'NO2 1st Max Value','NO2 1st Max Hour','O3 Mean','O3 1st Max Value','O3 1st Max Hour',
              'SO2 Mean','SO2 1st Max Value','SO2 1st Max Hour','CO Mean','CO 1st Max Value','CO 1st Max Hour'],axis=1)


# In[7]:


df.dropna(how="all",axis=1,inplace=True)
df.dropna(how="all",axis=0,inplace=True)


# In[9]:


df = df.dropna()
df = df[:9357]
df.rename(columns = {'NO2 AQI':'NO2AQI','SO2 AQI':'SO2AQI','O3 AQI':'O3AQI','CO AQI':'COAQI'},inplace = True)
i = 1
temp = []
while i <= 37421:
    try:
        #print(max (df.SO2AQI[i] , df.NO2AQI[i] , df.O3AQI[i] , df.COAQI[i]) )
        temp.append(max (df.SO2AQI[i] , df.NO2AQI[i] , df.O3AQI[i] , df.COAQI[i]))
    except :
        pass
    i += 1


df['aqi'] = temp
X = df.iloc[:,:-1]
#y = df['aqi']
y = df.iloc[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=0)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
score_rf = r2_score(y_test, y_test_pred)
#-------------------------------------------------model_code------------------------------------------------------------
#-------------
#
# -----------------------------------database---------------------------------------------------------------
import sqlite3
conn = sqlite3.connect('air_quality_database')
cur = conn.cursor()
try:
   cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL
   )''')

except:
   pass
#------------------------------------------------database---------------------------------------------------------------

from flask import Flask,render_template, url_for,request, flash, redirect, session
app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'

#-------------------------------------home_page-------------------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
   if not session.get('logged_in'):
      return render_template('home.html')
   else:
      return redirect(url_for('user_account'))

@app.route('/home1')
def home1():
   if not session.get('logged_in'):
      return render_template('home1.html')
   else:
      return redirect(url_for('user_account'))

#-------------------------------------home_page-------------------------------------------------------------------------

#-------------------------------------about_page-------------------------------------------------------------------------
@app.route("/about")
def about():
   return render_template('about.html')
#-------------------------------------about_page-------------------------------------------------------------------------

#-------------------------------------user_login_page-------------------------------------------------------------------------
@app.route('/user_login',methods = ['POST', 'GET'])
def user_login():
   conn = sqlite3.connect('air_quality_database')
   cur = conn.cursor()
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['psw']
      print('asd')
      count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
      print(count)
      #conn.commit()
      #cur.close()
      l = len(cur.fetchall())
      if l > 0:
         flash( f'Successfully Logged in' )
         return render_template('user_account.html')
      else:
         print('hello')
         flash( f'Invalid Email and Password!' )
   return render_template('user_login.html')

# -------------------------------------user_login_page-----------------------------------------------------------------

# -----------------------------------predict_page-----------------------------------------------------------------

@app.route('/predict', methods=['POST', 'GET'])
def predict():

    State = request.form['StateCode']
    Country = request.form['Country']
    NO2AQI = request.form['NO2AQI']
    O3AQI = request.form['O3AQI']
    SO2AQI = request.form['SO2AQI']
    COAQI = request.form['COAQI']
    global forest
    if request.method == 'POST':
       my_prediction = forest.predict([[float(State),float(Country),float(NO2AQI),float(O3AQI),float(SO2AQI),float(COAQI)]])
       flash('Air Quality is {}'.format( my_prediction[0]))

    return render_template('user_account.html')
# ------------------------------------predict_page-----------------------------------------------------------------

# ------------------------------------search_page-----------------------------------------------------------------
@app.route('/search')
def search():
   return render_template('search.html')
# ------------------------------------search_page-----------------------------------------------------------------

# -------------------------------------user_register_page-------------------------------------------------------------------------

@ app.route('/user_register', methods=['POST', 'GET'])
def user_register():
   conn = sqlite3.connect('air_quality_database')
   cur = conn.cursor()
   if request.method == 'POST':
      name = request.form['uname']
      email = request.form['email']
      password = request.form['psw']
      gender = request.form['gender']
      age = request.form['age']

      cur.execute("insert into user(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (name, email, password, gender, age))
      conn.commit()
      # cur.close()
      print('data inserted')
      return redirect(url_for('user_login'))

   return render_template('user_register.html')
# -------------------------------------user_register_page-------------------------------------------------------------------------

# -------------------------------------user_account_page-------------------------------------------------------------------------
@app.route('/user_account',methods = ['POST', 'GET'])
def user_account():
   return render_template('user_account.html')
# -------------------------------------user_account_page-------------------------------------------------------------------------

# -------------------------------------user_logout_page-------------------------------------------------------------------------
@app.route("/logout")
def logout():
   session['logged_in'] = False
   return home()

@app.route("/logoutd",methods = ['POST','GET'])
def logoutd():
   return home()# -------------------------------------user_logout_page-------------------------------------------------------------------------


if __name__ == '__main__':
   app.secret_key = os.urandom(12)
   app.run(debug=True)

