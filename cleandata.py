import pandas as pd

def clean_data():



    '''
    Reads in the raw csv file from kaggle with information about an nba player
    '''
    df = pd.read_csv("all_seasons 2.csv")


    '''
    Removes the columns of the descriptive information that don't seem as relevant like the college
    country, team name and index in the dataset
    '''
    df = df.drop(["draft_round" , "college" , "country" , "team_abbreviation","Unnamed: 0"],axis = 1)


    '''
    Season is in string format like "1996-1997" so it gets converted to int like 1996 with int conversion 
    and by splitting the string
    '''
    df["season"] = df["season"].apply(lambda x: int(x.split("-")[0]))



    #Initalizess player_map
    player_map = {}


    '''
    Loops through the whole dataframe and keeps mapping player_name to the current season
    Since the dataframe is chronological order, this will map each player name to the last season
    they were in the nba
    '''
    for i in range(0,len(df)):
        name = df.iloc[i]["player_name"]

        player_map[name] = [df.iloc[i]["season"]]


    '''
    For each data point, creates a boolean column in the dataframe called survived that measures if 
    a player was in the league five years later. It takes a player's last season
    from the player_map and measures if it's at least five greater than the current season
    '''
    df["last"] = df["player_name"].apply(lambda x:player_map[x][0])

    df["season_5"] = df["season"].apply(lambda x: x + 5)

    df["survived"] = (df["season_5"]<=df["last"])



    '''Helper functions'''

    '''For the position calculation 0-4 is supposed to go from pg to center(basketball positions)
    Sometimes the position formula is not perfect so it assigns position values less than zero to 0 and position
    values greater than 4 to 4
    '''

    def position(x):
        if(x<0):
            return 0
        elif(x>4):
            return 4
        else:
            return x


    '''
    Usually, players get picked 1st to 60th in the nba draft out of but some players 
    make it the nba without getting drafted so this function assigns the undrafted to draft pick
    100 to maintain consistent integer values throughout
    '''
    def draft_number(x):
        if x=="Undrafted":
            return 100
        else:
            return int(x)

    '''Applies draft_number helper function to the column using a lambda'''
    df["draft_number"] = df["draft_number"].apply(lambda x:draft_number(x))

    '''Uses an approximation of basketball statistics basketball-reference.com to create a picture of a 
    player's position 0-4 and then uses the position helper function to smooth the values'''
    df["position"] = (0.78 + 21.8*df["oreb_pct"] + -4.4*df["ast_pct"]).apply(lambda x:position(x))



    '''Approximate the number of possessions(how many shots were taken) while
    the player was on the court to help quantify his impact'''
    df["possessions"] = (2*df["reb"]/(df["oreb_pct"] + df["dreb_pct"]))













    #Long  regression estimating a player's offensive impact per 100 possessions using forumla on basketball-reference.com
    df["impact"] = (-0.33-0.036*df["position"])*(df["pts"])/(2*df["ts_pct"]) + (0.606*df["pts"])+ 0.476*df["ast"] - (0.5+0.2*df["position"])*df["ast"]*(0.584+0.075*df["position"]) + (0.203-0.02*df["position"])*df["reb"]*((df["oreb_pct"]+1)/(df["oreb_pct"] + df["dreb_pct"]+1))+ (-0.112+0.05*df["position"])*df["reb"]*((df["dreb_pct"]+1)/(df["oreb_pct"] + df["dreb_pct"]+1))

    df["impact"] = -4.208 + df["impact"]*100/df["possessions"]
    df["impact"].describe()


    '''Keeps the player_name and season and then index the dataframe to player_name and season'''
    df[["player" , "year"]] = df[["player_name" , "season"]]

    df = df.set_index(["player_name","season"])





    '''Collecting the player's impact five years from now into a list and returning null if the player
    is not the league five seasons in the future. It uses impacts map  
    to match a tuple of (player_name, year) to impact five years later if they survived. All the tuples match the index of the dataframe
    '''
    impacts={}
    for i in range(len(df)):
        if df.iloc[i]["survived"]:
                try:
                    impact = df.loc[(df.iloc[i]["player"], df.iloc[i]["year"]+5)]["impact"].iloc[0]

                except:

                    impact = df.loc[(df.iloc[i]["player"], df.iloc[i]["last"])]["impact"].iloc[0]

        else:
            impact = None
        impacts[(df.iloc[i]["player"], df.iloc[i]["year"])] = impact



    #Creates a column of the index

    df["index"] = df.index

    #Using the column of index it then creates a column matching to the impact of a player's impact five years from now 


    df["future impact"] = df["index"].apply(lambda x: impacts.get(x))













    #Drops helper columns
    df = df.drop(["season_5" , "draft_year" , "possessions","last","index"] , axis = 1)


    #Save the dataframe to a csv file
    df.to_csv("cleandata.csv")















































