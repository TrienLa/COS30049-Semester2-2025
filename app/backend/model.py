import pickle
import sklearn
import pandas

class NaiveBayes:
    def __init__(self):
        # Load data from pickle file
        with open('nb_model.pkl', 'rb') as tm:
            self.new_pipe = pickle.load(tm)
        pass
        

    def predict(self, input):
        # Make predictions
        results = self.new_pipe.predict(input)
        return results
    
class LinearRegression:
    def __init__(self):
        # Load data from pickle file
        with open('lr_model.pkl', 'rb') as tm:
            self.new_pipe = pickle.load(tm)
        pass
        

    def predict(self, input):
        # Make predictions
        results = self.new_pipe.predict(input)
        return results