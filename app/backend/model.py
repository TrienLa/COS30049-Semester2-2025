import pickle
import pandas as pd
from data_utils import data_clean_up, data_preprocessing, load_data

class NaiveBayes:
    def __init__(self):
        # Load data from pickle file
        with open('nb_model.pkl', 'rb') as tm:
            self.new_pipe = pickle.load(tm)
        pass
        
    def predict(self, input):
        # Turn input data into Dataframe for clean up and pre-processing
        df = data_clean_up(data_preprocessing(input))

        # Make predictions
        results = self.new_pipe.predict(df['text'])
        return results
    
class LinearRegression:
    def __init__(self):
        # Load data from pickle file
        with open('lr_model.pkl', 'rb') as tm:
            self.new_pipe = pickle.load(tm)
        pass
        
    def predict(self, input):
        # Turn input data into Dataframe for clean up and pre-processing
        df = data_clean_up(data_preprocessing(input))

        # Make predictions
        results = self.new_pipe.predict(df['text'])
        return results

class SpamClassifier:
    def __init__(self):
        self.models = {
            'NaiveBayes': NaiveBayes(),
            'LinearRegression': LinearRegression()
        }
    
    def spam_classify(self, csv_file, model_name='NaiveBayes'):
        """
        Process CSV file with email data and return predictions
        
        Expected CSV format:
        title,text
        "Subject 1","Email content 1"
        "Subject 2","Email content 2"
        """
        try:
            # Read CSV file
            df = load_data(csv_file)
            
            # Validate required columns
            if 'text' not in df.columns:
                raise ValueError("CSV must contain 'text' column")
            
            # Get the selected model
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Make predictions
            predictions = model.predict(df)
        
            # Format results
            results = []
            valid_count = 0
            spam_count = 0
            
            for (index, row), prediction in zip(df.iterrows(), predictions):
                prediction_label = 'spam' if prediction == 1 else 'ham'
                
                if prediction_label == 'ham':
                    valid_count += 1
                else:
                    spam_count += 1
                
                results.append({
                    'title': row.get('title', f'Message {index + 1}'),
                    'text': row.get('text', ''),
                    'prediction': prediction_label
                })

            return {
                'predictions': results,
                'valid_count': valid_count,
                'spam_count': spam_count,
                'total_count': len(results)
            }
            
        except Exception as e:
            raise Exception(f"Error processing CSV: {str(e)}")