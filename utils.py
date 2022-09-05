import numpy as np
import pickle
import os

file_path = r'iris_classification/knn_reg_model.pkl'
file_path2= r'iris_classification/normal_scaler.pkl'


class IrisClassification():
    def __init__(self, sepal_length,sepal_width,petal_length,petal_width):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def get_model(self):
        with open( "knn_reg_model.pkl",'rb') as f:
            self.model = pickle.load(f)
    def get_scale(self):
        with open( "normal_scaler.pkl",'rb') as f:
            self.scaler_model = pickle.load(f)
    
    
    def get_predicted_class(self):
        self.get_model()
        self.get_scale()
        input_array = np.array([self.sepal_length,self.sepal_width,
                                self.petal_length,self.petal_width],ndmin = 2)
        print(input_array)
        
        scaled_data =self.scaler_model.transform(input_array)
        print(scaled_data)
        predicted_class = self.model.predict(scaled_data)[0]
        
        classes = { 0 : "Iris-Setosa", 
                    1 : "Iris-Versicolour",
                    2 : "Iris-Virginica" }

        result = classes[predicted_class]
        return result

if __name__ == "__main__":
    Obj = IrisClassification(3,4,5,1)
    # Obj.get_predicted_class()
    result = Obj.get_predicted_class()
    print("Predicted CLass is :",result)
