import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load model & scaler
# ----------------------------
model = pickle.load(open("ipl_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ----------------------------
# Define Encodings (must match training)
# ----------------------------
teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants",
    "Gujarat Titans"
]

venues = [
    "Wankhede Stadium", "Eden Gardens", "M. A. Chidambaram Stadium",
    "Arun Jaitley Stadium", "M. Chinnaswamy Stadium"
]

win_types = ["runs", "wickets", "super_over"]

team_encoding = {team: idx for idx, team in enumerate(teams)}
venue_encoding = {venue: idx for idx, venue in enumerate(venues)}
toss_decision_encoding = {"bat": 0, "field": 1}
win_type_encoding = {wt: idx for idx, wt in enumerate(win_types)}

# ----------------------------
# App Title
# ----------------------------
st.title("🏏 IPL Match Winner Predictor")
st.markdown("### Enter match details to predict the winner")

# ----------------------------
# User Inputs
# ----------------------------
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.radio("Toss Decision", ["bat", "field"])
win_type = st.selectbox("Win Type", win_types)

win_margin = st.number_input("Win Margin (runs/wickets)", 0, 200, 20)
first_innings = st.number_input("First Innings Score", 50, 300, 160)
second_innings = st.number_input("Second Innings Score", 50, 300, 150)
powerplay = st.number_input("Powerplay Score", 0, 100, 50)
middle_overs = st.number_input("Middle Overs Score", 0, 120, 70)
death_overs = st.number_input("Death Overs Score", 0, 80, 40)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Winner"):

    # order must match training columns
    features = np.array([[
        team_encoding[team1],         # Teams
        venue_encoding[venue],        # Venue
        team_encoding[toss_winner],   # Toss_Winner
        toss_decision_encoding[toss_decision],  # Toss_Decision
        win_type_encoding[win_type],  # Win_Type
        win_margin,                   # Win_Margin
        first_innings,                # First_Innings_Score
        second_innings,               # Second_Innings_Score
        powerplay,                    # Powerplay_Scores
        middle_overs,                 # Middle_Overs_Scores
        death_overs                   # Death_Overs_Scores
    ]])

    # scale with the same scaler as training
    features_scaled = scaler.transform(features)

    # predict
    prediction = model.predict(features_scaled)[0]

    # here we assume label 1 = team1, 0 = team2 (adjust if opposite)
    winner = team1 if prediction == 1 else team2

    st.success(f"🏆 Predicted Winner: **{winner}**")
