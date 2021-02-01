# nba

Dataset Link: https://www.kaggle.com/justinas/nba-players-data

Libraries used:
Sklearn:https://scikit-learn.org/stable/
Helps train the model and print accuracy metrics from build_model.py



Pandas:https://pandas.pydata.org/docs/
Helps manipulate data from csv files

Joblib: https://joblib.readthedocs.io/en/latest/

Helps save the models and upload the models for later use

The  converts the kaggle dataset into a csv
file that can be used to train the random forest models.

The build_model saves the model that predicts a player's offensive impact five years from now into future_impact_model.sav and 
saves the random forest classifier that predicts whether or not an nba player will still be in the league five years from now into survival_classification.sav


The nbaoutput.py file has the main function that runs clean_data() and build_model()
It also outputs some sample predictions from both models for some famous nba players.





#Running the program

'''

$git clone https://github.com/java-final-project-ml/nba

$cd nba

$pip install requirements.txt

$python nbaoutput.py
'''







