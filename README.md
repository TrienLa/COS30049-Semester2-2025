# Email Classification Models
## Overview
This project aims to train on a dataset of various spam emails and SNS messages to utilize a machine learning algorithm ultimately classify harmful contents feed back to the model. The project mainly use Python machine learning library `sklearn` to generate and train these classification models.

## Setting up
To create your local copy of the file, you can use this command in a command prompt. This will download all the repository files onto your device.
```
git clone https://github.com/TrienLa/COS30049-Semester2-2025
```

## Environment and dependencies
Once you download / clone the repository, you can start setting up the environment to run the project. Run the following command after you CD to the project directory to install any missing prerequisite libraries. 
```
python install -r requirements.txt
```

## Dataset Cleanup & Preparation
The dataset we have will need further processing, so running ` data_processing.py ` to clean up the dataset while also extracting features for our models' training.
```
python data_processing.py
```

## Model Generation & Output
Now that we have both the environment and the dataset, we can start running ` nb_model.py `, ` lregression_model.py `, ` kmeans_model.py ` to generate their corresponding models which we can retrieve at a later point to do real-time classification.
```
python nb_model.py
python lregression_model.py
python kmeans_model.py
```

We use Python's ` pickle ` library to serialise each model and load them later when we need them.
