from flask import Flask, jsonify,redirect,request
import config
from utils import IrisClassification
app = Flask(__name__)




@app.route('/')
def hello_flask():
    print('*'*90)
    print("We are testing Flask")
    print('*'*90)
    return jsonify({"Message" : "Welcome to Flask"})









##############################################################################
######################## Prediction ##########################################
##############################################################################
@app.route('/prediction',methods = ['POST','GET'])
def prediction():

    input_data = request.form
    sepal_length = float(input_data['sepal_length'])
    sepal_width = float(input_data['sepal_width'])
    petal_length = float(input_data['petal_length'])
    petal_width = float(input_data['petal_width'])

    Obj = IrisClassification(sepal_length,sepal_width,petal_length,petal_width)
    result = Obj.get_predicted_class()
    print("Predicted CLass is :",result)

    return jsonify({"Result":f"Predicted Class is : {result}"})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = config.PORT_NUMBER, debug=False)