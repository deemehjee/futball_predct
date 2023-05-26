#!/usr/bin/env python
# coding: utf-8

# # football predict program using python 

# In[ ]:





# In[ ]:



import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
from sklearn.linear_model import Lasso

# Function to retrieve match data from the API
def get_match_data(api_url):
    
    def get_match_data(api_url):
    response = requests.get(api_url)
    data = response.json()
    match_data = []

    for match in data['matches']:
        home_team = match['home_team']
        away_team = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        tackles = match['tackles']  # Replace with actual data for tackles
        fouls_conceded = match['fouls_conceded']  # Replace with actual data for fouls conceded
        interceptions = match['interceptions']  # Replace with actual data for interceptions
        passes = match['passes']  # Replace with actual data for passes
        crosses = match['crosses']  # Replace with actual data for crosses
        corners = match['corners']  # Replace with actual data for corners
        
        match_data.append([home_team, away_team, home_goals, away_goals, tackles, fouls_conceded, interceptions, passes, crosses, corners])

    return match_data

# Get user input for API URL
api_url = input("Enter the API URL: e.g https://api.betfair.com/sports/football/matches?season=2020-2023")

# Retrieve match data from the API
match_data = get_match_data(api_url)

# Convert match data to a pandas DataFrame
epl_data = pd.DataFrame(match_data, columns=['HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'Tackles', 'FoulsConceded', 'Interceptions', 'Passes', 'Crosses', 'Corners'])

# Data preprocessing
epl_data = epl_data[:-10]  # Exclude last 10 matches

# Get user input for home team and away team
home_team = input("Enter the home team: ").lower()
away_team = input("Enter the away team: ").lower()

# Filter the data for the specific home team and away team
team_data = epl_data[(epl_data['HomeTeam'].str.lower() == home_team) & (epl_data['AwayTeam'].str.lower() == away_team)]

# Check if the provided teams have sufficient historical data
if len(team_data) < 1:
    print("Insufficient historical data for the provided teams.")
else:
    # Extract the relevant features and target variable
    X = team_data[['Tackles', 'FoulsConceded', 'Interceptions', 'Passes', 'Crosses', 'Corners']].values
    y_goals = team_data[['HomeGoals', 'AwayGoals']].values
    y_red_cards = team_data['RedCards'].values
    y_yellow_cards = team_data['YellowCards'].values
    y_shots = team_data['Shots'].values

    # Split the data into training and testing sets
    split_idx = len(team_data) - 1
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_goals_train, y_goals_test = y_goals[:-1], y_goals[-1:]
    y_red_cards_train, y_red_cards_test = y_red_cards[:-1], y_red_cards[-1:]
    y_yellow_cards_train, y_yellow_cards_test = y_yellow_cards[:-1], y_yellow_cards[-1:]
    y_shots_train, y_shots_test = y_shots[:-1], y_shots[-1:]

    # Train the Lasso regression model for goals prediction
    goals_model = Lasso()
    goals_model.fit(X_train, y_goals_train)

    # Train the Lasso regression model for red cards prediction
    red_cards_model = Lasso()
    red_cards_model.fit(X_train, y_red_cards_train)

    # Train the Lasso regression model for yellow cards prediction
    yellow_cards_model = Lasso()
    yellow_cards_model.fit(X_train, y_yellow_cards_train)

    # Train the Lasso regression model for shots prediction
    shots_model = Lasso()
    shots_model.fit(X_train, y_shots_train)

    # Make predictions
    goals_prediction = goals_model.predict(X_test)[0]
    red_cards_prediction = red_cards_model.predict(X_test)[0]
    yellow_cards_prediction = yellow_cards_model.predict(X_test)[0]
    shots_prediction = shots_model.predict(X_test)[0]
    corners_prediction = X_test[0][5]  # Use the actual corners value as prediction

# team_data['Corners'].values[-1]

    # Print the predictions
    print("Predictions for", home_team.capitalize(), "vs", away_team.capitalize())
    print("Goals prediction:", goals_prediction)
    print("Red cards prediction:", red_cards_prediction)
    print("Yellow cards prediction:", yellow_cards_prediction)
    print("Shots prediction:", shots_prediction)
    print("Corners prediction:", corners_prediction)



# In[ ]:





# In[ ]:





# In[1]:


#Note that you need to replace the placeholders (tackles, fouls_conceded, interceptions, passes, crosses, corners) 
#in the get_match_data function with the actual data obtained from the Betfair API. 
#Also, make sure to import the necessary libraries (mean_squared_error) and 
#adjust the feature names (Tackles, FoulsConceded, Interceptions, Passes, Crosses, Corners) based on the available data.


# In[ ]:




