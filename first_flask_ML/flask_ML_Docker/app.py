#import Flask 
from flask import Flask, request, render_template
import joblib

#create an instance of Flask
app = Flask(__name__)

@app.route('/')
@app.route('/index')

def home():
    return render_template('index.html')

@app.route('/result', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
       input_list = request.form.values()
       input_list = list(map(float, input_list))
       print(input_list)

       model_load = joblib.load('model.pkl')
       prediction = model_load.predict([input_list])[0]

       return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0')