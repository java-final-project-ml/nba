
'''Import the pandas library for data manipulation'''
import pandas as pd

def build_model():




  #Read in the cleaned dataset
  df = pd.read_csv("cleandata.csv")



  #We don't have the statistics of players five years in the future
  df = df[df["season"]<2014].reset_index()


  '''Some players did not get drafted into the nba out of college so we just assign them all as being the "100th pick"
  '''
  df = df.fillna(100)




  '''
  First dataset is the dataset of the player's in any given season who are in the league five years later to predict future impact from current impact
  For this, we need to only get the subset of players that survived
  I set to nonly players that have played 15 games because below that it's too small of a sample size
  '''
  survived = df[df["survived"]==True].reset_index()

  survived = survived[survived["gp"]>15]

  '''Removes all the extra columns that are not relevant independent variables'''
  extra_cols = ["survived" , "future impact","season","player_name","index" , "level_0" , "year" , "player"] 
  survived_features = survived.drop(extra_cols , axis = 1)

  #survived_features =survived_features[["player_height" , "player_weight" , "age" , "draft_number" , "impact" , "gp"]]
  '''Creates the dataframe of the dependent variable future impact"'''
  survived_impact = survived[["future impact"]]


  '''This drops all the Removes all the extra columns that are not relevant independent variables but takes the full
  dataset because this is for the classifier that predicts whether or not a player will be in the nba five year from now so it needs both survivors
  and non-survivors'''
  full_features = df.drop(["survived" , "future impact","season","player_name","index","year","player"],axis = 1)


  '''This takes the survival as the dependent variable for the binary classifier'''
  full_survival = df[["survived"]]


  '''Converts the dataframe into the array format that the SkLearn library requiress'''
  x = survived_features.values

  y = survived_impact.values

  x_full= full_features.values

  y_full = full_survival.values


  '''Splits the data into training and test set, imports RandomForest algorithm from SkLearn
  and fits the survival classifier and the future impact regressor to the data'''
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.11, random_state = 11)

  x_full_train, x_full_test, y_full_train, y_full_test = train_test_split(x_full, y_full, test_size = 0.22, random_state = 11)
  #fitting Random forest regression model to the dataset
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.ensemble import RandomForestClassifier
  regressor = RandomForestRegressor(n_estimators =77, random_state = 11,max_features = 7 , max_depth = 7)

  classifier = RandomForestClassifier(n_estimators = 222, random_state = 11)
  regressor.fit(x,y)

  classifier.fit(x_full_train , y_full_train)


  '''Prints the metrics of model performance'''

  from sklearn.metrics import confusion_matrix

  cm= confusion_matrix(y_full_test, classifier.predict(x_full_test))

  acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0]+cm[0][0] + cm[1][1])
  print("Accuracy is " + "rounded to" + str(int(acc*100)) + "%" + " when predicting whether or not a player will be in the league 5 years from now")





  y_pred = regressor.predict(x)

  y_pred = pd.Series(y_pred)
  y_actual = pd.Series(y.ravel())

  corr = y_pred.corr(y_actual)

  print("the correlation between predicted future impact and real future impact is " + str(float(corr))[:5])





  '''Using joblib the future impact model and the surival are saved'''
  filename = 'future_impact_model.sav'

  joblib.dump(regressor, filename)

  filename = "survival_classification.sav"

  joblib.dump(classifier , filename)








































