import streamlit as st
import requests
import pandas as pd

# ==========================
# --- STREAMLIT SIDEBAR ---
# ==========================
st.set_page_config(page_title="Top-Down Betting Dashboard", layout="wide")

st.sidebar.header("Dashboard Settings")

# User inputs
api_key = st.sidebar.text_input("Enter The Odds API Key", type="password")
all_sports_list = ["basketball_nba", "football_nfl", "baseball_mlb", "soccer_epl"]
selected_sports = st.sidebar.multiselect("Select Sport(s)", options=all_sports_list)
market_category = st.sidebar.selectbox("Market Category", ["game_lines", "player_props"])
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1, value=1000)
min_edge = st.sidebar.slider("Minimum Edge (%)", 0, 100, 0)
min_roi = st.sidebar.slider("Minimum ROI (%)", -100, 100, -100)

st.sidebar.markdown("---")

# ==========================
# --- FUNCTIONS ---
# ==========================

def fetch_odds(api_key, sport_keys):
    """Fetch odds from The Odds API for the selected sports."""
    all_games = []
    headers = {"Accept": "application/json"}
    for sport in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?regions=us&markets=moneyline,spreads,totals,player_points,player_assists,player_rebounds&apiKey={api_key}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            all_games.extend(response.json())
        else:
            st.error(f"Failed to fetch odds for {sport}. Status code: {response.status_code}")
    return all_games

def get_available_player_props(data, sport_key):
    """Return sorted list of available player prop markets for a sport."""
    available_props = set()
    for game in data:
        if game.get('sport_key') != sport_key:
            continue
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key', '').startswith("player_"):
                    available_props.add(market['key'])
    return sorted(list(available_props))

def get_available_game_markets(data, sport_key):
    """Return sorted list of game line markets (moneyline, spread, totals)."""
    game_markets = set()
    for game in data:
        if game.get('sport_key') != sport_key:
            continue
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') in ["spreads", "totals", "moneyline"]:
                    game_markets.add(market['key'])
    return sorted(list(game_markets))

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def implied_probability(decimal_odds):
    """Calculate implied probability from decimal odds."""
    return 1 / decimal_odds

def remove_vig(prob_list):
    """Normalize probabilities to remove bookmaker vig."""
    total = sum(prob_list)
    return [p/total for p in prob_list]

def calc_edge(implied_prob, predicted_prob):
    """Edge = (predicted_prob - implied_prob) / implied_prob"""
    return (predicted_prob - implied_prob) / implied_prob * 100  # percentage

def calc_ev(decimal_odds, predicted_prob):
    """Expected Value per $1 bet."""
    return (decimal_odds * predicted_prob) - 1

def calc_kelly(decimal_odds, predicted_prob, bankroll):
    """Kelly stake recommendation in dollars."""
    b = decimal_odds - 1
    p = predicted_prob
    q = 1 - p
    fraction = (b * p - q) / b
    return max(fraction * bankroll, 0)

# ==========================
# --- FETCH ODDS ---
# ==========================
if st.sidebar.button("Fetch Live Odds"):
    if not api_key:
        st.warning("Please enter your API key.")
    elif not selected_sports:
        st.warning("Please select at least one sport.")
    else:
        data = fetch_odds(api_key, selected_sports)
        if not data:
            st.error("No odds returned from API.")
        else:
            st.success(f"Fetched odds for {len(selected_sports)} sport(s).")

            # ==========================
            # --- MARKET DROPDOWNS ---
            # ==========================
            market_selection = {}
            if market_category == "player_props":
                for sport_key in selected_sports:
                    props_list = get_available_player_props(data, sport_key)
                    if props_list:
                        market_selection[sport_key] = st.selectbox(
                            f"Select Player Prop Type for {sport_key}",
                            props_list,
                            key=f"{sport_key}_prop"
                        )
                    else:
                        st.warning(f"No player prop types available for {sport_key}")
                        market_selection[sport_key] = None

            elif market_category == "game_lines":
                for sport_key in selected_sports:
                    game_markets = get_available_game_markets(data, sport_key)
                    if game_markets:
                        market_selection[sport_key] = st.selectbox(
                            f"Select Game Market for {sport_key}",
                            game_markets,
                            key=f"{sport_key}_game"
                        )
                    else:
                        st.warning(f"No game line markets available for {sport_key}")
                        market_selection[sport_key] = None

            # ==========================
            # --- BUILD ODDS TABLE WITH CALCULATIONS ---
            # ==========================
            table_rows = []

            # Example predicted probability (placeholder for ML model)
            # Replace this with your real ML model predictions
            example_predicted_prob = 0.55  # 55%

            for game in data:
                sport = game.get("sport_key")
                home_team = game.get("home_team")
                away_team = game.get("away_team")
                commence_time = game.get("commence_time")
                
                for bookmaker in game.get("bookmakers", []):
                    book_name = bookmaker.get("title")
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key")
                        for outcome in market.get("outcomes", []):
                            odds = outcome.get("price")
                            decimal_odds = american_to_decimal(odds)
                            implied_prob = implied_probability(decimal_odds)
                            ev = calc_ev(decimal_odds, example_predicted_prob)
                            edge = calc_edge(implied_prob, example_predicted_prob)
                            roi = ev * 100  # % ROI
                            kelly = calc_kelly(decimal_odds, example_predicted_prob, bankroll)
                            
                            # Filter by thresholds
                            if edge < min_edge or roi < min_roi:
                                continue
                            
                            row = {
                                "Sport": sport,
                                "Game": f"{away_team} @ {home_team}",
                                "Commence Time": commence_time,
                                "Bookmaker": book_name,
                                "Market": market_key,
                                "Outcome": outcome.get("name"),
                                "American Odds": odds,
                                "Decimal Odds": round(decimal_odds, 2),
                                "Implied Prob": round(implied_prob, 3),
                                "Predicted Prob": example_predicted_prob,
                                "Edge (%)": round(edge, 2),
                                "EV ($1)": round(ev, 3),
                                "ROI (%)": round(roi, 2),
                                "Kelly Stake ($)": round(kelly, 2)
                            }
                            table_rows.append(row)

            if table_rows:
                df = pd.DataFrame(table_rows)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download Odds CSV",
                    df.to_csv(index=False),
                    file_name="live_odds_with_ev.csv",
                    mime="text/csv"
                )
            else:
                st.info("No odds passed the filters for the selected options.")
