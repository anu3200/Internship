import  numpy as np
import pickle
from flask import Flask, render_template, request
model=pickle.load(open('salary_pred_model.pkl','rb'))
label=pickle.load(open('label.pkl','rb'))
app=Flask(__name__)
@app.route('/')  
def home():
    return render_template('home.html')   
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    age=request.form['age']
    workclass=request.form['workclass']
    education=request.form['education']
    maritalstatus=request.form['maritalstatus']
   
    occupation=request.form['occupation']
    relationship=request.form['relationship']
    race=request.form['race']
    sex=request.form['sex']
   
    hoursperweek=request.form['hoursperweek']
    nativecountry=request.form['nativecountry']
    
    values=[age,workclass,education,maritalstatus,occupation,relationship,race,
            sex,hoursperweek,nativecountry]
    data=[]
    for i in values:
        data.append(i)
    data = label.fit_transform(data)
    
    result=np.array(data).reshape(1,-1)
    prediction=model.predict(result)
    if(prediction[0]==0):
        output="less than or equal to 50k USD"
    else:
        output="greater than 50k USD"
    
    #output=prediction.item()
    
    return render_template('result.html',prediction_text=format(output))


if __name__ == "__main__":                            
    app.run()
    
