from flask import Flask
from flask import render_template,request,jsonify,send_file
from flask_cors import CORS
from flask_mysqldb import MySQL
import os

# importing the ML model
import model as md
# Create a folder uploads
UPLOAD_FOLDER = 'uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# configuring mysql
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Mysql26!'
app.config['MYSQL_DB'] = 'login'
# Enabling Cross Origin Resource Sharing(CORS) inorder to receive data from another domain (Angular)
CORS(app)
# Creating an instance for the mysql
mysql = MySQL(app)

# signup
@app.route("/signup",methods=['POST'])
def signup():
 try:    
    username = request.form.get('username');
    password = request.form.get('password');
    email = request.form.get('email')
    cur = mysql.connection.cursor() #establishing connection    
    #cursor is used to execute queries and fetch data from db
    

    # checking if the user name already exists in the db
    if(findUserByName(username) is not None): 
       statusCode = 409 #Conflict: user already exists
       response = {
          "statusCode":statusCode,
          "responseMessage":"User Already Exists"
       }
       return response,statusCode
    else:
       statusCode = 200
       response = {
          "statusCode":statusCode,
          "responseMessage":"Sign Up Successful"
       }
    # Registering a new user
    cur.execute("INSERT INTO users(username, password,mail) VALUES (%s, %s,%s)", (username, password,email))
    mysql.connection.commit()
    cur.close()
    return jsonify(response),statusCode
 except Exception as e:
    print(e)
    return jsonify({
       "statusCode":500,
       "responseMessage":"Internal Server Error"
    }),500;


@app.post("/login")
def login():
   try:    
    username = request.form.get('username');
    password = request.form.get('password');
    cur = mysql.connection.cursor() #establishing connection    
    #cursor is used to execute queries and fetch data from db

    user = findUserByName(username)
    if(user):
       if(user[0][1]==password): 
          statusCode = 200
          response = {
             "statusCode":statusCode,
             "responseMessage":"Login Successful"
          }
       else: 
          statusCode = 401 #Unauthorized
          response = {
             "statusCode":statusCode,
             "responseMessage":"Incorrect Password"
          }
    else:
       statusCode = 401 #Unauthorized
       response = {
          "statusCode":statusCode,
          "responseMessage":"User Does not exist"
       }

    mysql.connection.commit()
    cur.close()
    return jsonify(response),statusCode
   except Exception as e:
    print(e)
    return jsonify({
       "statusCode":500,
       "responseMessage":"Internal Server Error"
    }),500;

@app.get('/logout')
def logout():
   try:
      # deleting the saved files when the user logs out
      os.remove(f"{UPLOAD_FOLDER}/dataset.csv")
      os.remove(f"{UPLOAD_FOLDER}/predictions.csv")
      
   except:
      # If file didnot exist in the first place
      pass
   return "Logged out",200

   

# getting form data
@app.post("/save")
def save():
    # req: is of dict type
    # req = request.get_json()

    print(request)
    print(request.form.get('periodicity'))
    print(request.form.get('num'))
    periodicity = request.form.get('periodicity')
    num = request.form.get('num')
    #Gives all the files in the request payload in the form of ImmutableMultiDict(provided by werkzeug)
    file = request.files['File']
    print(file.filename)
   
    # Saving the received file in the uploads folder with the name dataset.csv
    file.save(f"{UPLOAD_FOLDER}/dataset.csv")
   #  Building model
    md.main(periodicity,num)
    return "received"   


#Returns the user tuple by their user name, If user doesn't exist return none
def findUserByName(username):
    cur = mysql.connection.cursor() #establishing connection    
    #cursor is used to execute queries and fetch data from db

    # checking if the user name already exists in the db
    cur.execute("SELECT * FROM users WHERE(username=%s)",(username,))
    resultSet = cur.fetchall()
    if(resultSet!=()):  return resultSet
    return None

# return data set
@app.get("/predict")
def dataset():
   try:
      return md.datapts(),200;
   except Exception as e:
      return {
         "statusCode":500,
         "ErrorMessage":e
      },500

#Sending the predicted csv file to the user
@app.route('/get_csv')
def get_predicted_sales():
    filepath = 'uploads/predictions.csv'
    return send_file(filepath, as_attachment=True)