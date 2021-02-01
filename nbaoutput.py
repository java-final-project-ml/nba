#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:24:12 2021

@author: madhavrai
"""

import pandas as pd
import joblib

from build_model import build_model

from cleandata import clean_data

def main():
    clean_data()
    
    build_model()
    

    #Reads the csv built from "clean_data()" function called from the first file
    df = pd.read_csv("cleandata.csv")



    #CThen the dataframe is condensed down to data points from the last season
    df = df[df["season"]==2019]


    dependent = ["future impact" , "survived"]
    #This loop goes through and takes out any player with null independent variables
    for col in df.columns:
        if col not in dependent:
            df = df[df[col]!=None] 


    df = df.reset_index()

    #Condenses the dataframe to players that played at least 15 games last season to match the training dataset
    df = df[df["gp"]>15]

    #df = df.drop(["index"], axis = 1)








    #Initializes a dictionary of current players
    current = {}

    #Load the model that predicts if a player will survive five more years and the model that projects future impact
    classifier = joblib.load("survival_classification.sav")

    regressor = joblib.load("future_impact_model.sav")


    '''
    This for loop goes through the dataframe and maps a player's name to the prediction of whether or not the player will be in the league
    five years from now and the prediction of a player's projected impact five years from now
    '''
    for i in range(len(df)):
        #Columns that are not relevenat  independent variables are dropped
        features = df.iloc[i].drop(["survived" , "future impact","season","player_name","index","year","player"]).values.reshape(1, -1)
        '''Player name is matched to list containing the survival prediction from the classifier
            and the future impact 
        '''
        current[df.iloc[i]["player_name"]] = [classifier.predict(features), regressor.predict(features),(2*df["reb"].iloc[i]/(df["oreb_pct"].iloc[i] + df["dreb_pct"].iloc[i]))]







    #This function summarizes the prediction about a player in a nice readable format using the map current of current players
    def future(player):
        try:
            if current.get(player)[0]:
                future = player + " is expected to be in the league five years from now" + " and his offensive impact is expected to be " + str(current.get(player)[1][0]) + " points per 100 possessions  five years from now"
            else:
                future = player + " is not expected to be in the league five years from now"
        except:
            future = "This player did not play 15+ games in the 2019-2020 nba season"
        return future



    #Sets a list of some of the moust famous players in the nba
    famous_players = ["Kawhi Leonard","LeBron James","Paul George", "Kyrie Irving" , "Nikola Jokic" , "Luka Doncic" , "Lonzo Ball" , "Dwight Howard" , "Vince Carter"]


    #Loops through the list of famous players and prints the summarized prediction for all of them
    for player in famous_players:
        print(future(player))
        print("----------------------------------")
main()









