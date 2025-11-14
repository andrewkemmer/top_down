# app.py
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Sharp EV Top-Down Betting Dashboard")

# -----------------------------
# Helper Functions
# -----------------------------
def american_to_decimal(odds):
    odds = float(odds)
    return 1.0 + (odds / 100.0) if odds > 0 else 1.0 + (100.0 / -odds)

def american_to_prob(odds):
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)

def remove_vig_general(prob_map):
    """Normalize a dict of probabilities so they sum to 1."""
    total = sum(prob_map.values())
    if total <= 0:
        return prob_map.copy()
    return {k: v / total for k, v in prob_map.items()}

def kelly_fraction(true_p, decimal_odds):
    b = decimal_odds - 1.0
    # avoid division by zero
    if b <= 0:
        return 0.0
    f = (b * true_p - (1 - true_p)) / b
    return max(f, 0.0)

# -----------------------------
# TheOddsAPI helper
# -----------------------------
THEODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_odds_theoddsapi(api_key, sport, region="us", market="h2h"):
    url = f"{THEODDS_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": region,
        "markets": market,
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Prop markets dictionary (NBA, NFL, MLB, NHL)
# -----------------------------
prop_markets = {
    # NBA
    "NBA Player Points": "player_points",
    "NBA Player Assists": "player_assists",
    "NBA Player Rebounds": "player_rebounds",
    "NBA Player PAR (Points+Assists+Rebounds)": "player_points_assists_rebounds",
    "NBA Player 3-Pointers Made": "player_three_pointers_made",
    "NBA Player Steals": "player_steals",
    "NBA Player Blocks": "player_blocks",

    # NFL
    "NFL Player Passing Yards": "player_passing_yards",
    "NFL Player Rushing Yards": "player_rushing_yards",
    "NFL Player Receiving Yards": "player_receiving_yards",
    "NFL Player Receptions": "player_receptions",
    "NFL Player Rush Attempts": "player_rush_attempts",
    "NFL Player Pass Attempts": "player_pass_attempts",
    "NFL Player Touchdowns": "player_touchdowns",

    # MLB - Hitting
    "MLB Player Hits": "player_hits",
    "MLB Player Total Bases": "player_total_bases",
    "MLB Player Home Runs": "player_home_runs",
    "MLB Player RBIs": "player_rbis",
    "MLB Player Runs Scored": "player_runs",
    "MLB Player Singles": "player_singles",
    "MLB Player Doubles": "player_doubles",
    "MLB Player Triples": "player_triples",
    "MLB Player Walks": "player_walks",
    "MLB Batter Strikeouts": "player_strikeouts",
    # MLB - Pitching
    "MLB Pitcher Strikeouts": "pitcher_strikeouts",
    "MLB Pitcher Outs Recorded": "pitcher_outs",
    "MLB Pitcher Earned Runs": "pitcher_earned_runs",
    "MLB Pitcher Hits Allowed": "pitcher_hits_allowed",
    "MLB Pitcher Walks Allowed": "pitcher_walks",

    # NHL - Skaters
    "NHL Player Goals": "player_goals",
    "NHL Player Assists": "player_assists_nhl",
    "NHL Player Points": "player_points_nhl",
    "NHL Player Shots on Goal": "player_shots_on_goal",
    "NHL Player Shot Attempts": "player_shot_attempts",
    "NHL Player Blocked Shots": "player_blocked_shots",
    "NHL Player Penalty Minutes": "player_penalty_minutes",
    # NHL - Goalies
    "NHL Goalie Saves": "goalie_saves",
    "NHL Goalie Goals Allowed": "goalie_goals_allowed",
}

# -----------------------------
# Streamlit UI - Sidebar
# -----------------------------
st.title("🔎 Sharp EV Top-Down Betting Dashboard")

st.sidebar.header("API / Data Source")
api_key = st.sidebar.text_input("TheOddsAPI Key", value=os.getenv("THEODDS_API_KEY", ""))

# sports list fetch (if API key provided)
sport_options = {}
if api_key:
    try:
        sports_list = requests.get(f"{THEODDS_BASE}/sports/?apiKey={api_key}").json()
        sport_options = {s["title"]: s["key"] for s in sports_list}
    except Exception as e:
        st.sidebar.error(f"Error fetching sports list: {e}")

# default sports choices when API not available or fetch fails
default_sports = ["NBA", "MLB", "NHL", "NFL"]
selected_sports = st.sidebar.multiselect(
    "Select sport(s)",
    options=list(sport_options.keys()) if sport_options else default_sports,
    default=["MLB"] if "MLB" in (sport_options.keys() if sport_options else default_sports) else (list(sport_options.keys())[:1] if sport_options else ["NBA"])
)

region = st.sidebar.selectbox("Region", ["us", "uk", "eu", "au"], index=0)

# Market category (Game vs Player Props)
market_category = st.sidebar.radio("Market Category", ["Game Lines", "Player Props"], index=0)

# Game markets mapping
market_label_to_code = {
    "Moneyline (H2H)": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

if market_category == "Game Lines":
    market_selection_label = st.sidebar.selectbox("Market Type", list(market_label_to_code.keys()), index=0)
    market_code = market_label_to_code[market_selection_label]
else:
    # Player props: pick from prop_markets
    prop_label = st.sidebar.selectbox("Player Prop Type", options=list(prop_markets.keys()))
    market_code = prop_markets[prop_label]

# Kelly and filters
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1.0, value=1000.0, step=1.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
use_kelly = st.sidebar.checkbox("Show Kelly stake", value=True)

min_ev = st.sidebar.number_input("Min EV per $1", value=0.0)
min_edge = st.sidebar.number_input("Min edge", value=0.0)
min_roi = st.sidebar.number_input("Min ROI %", value=0.0)

sharp_book_key_input = st.sidebar.text_input("Sharp sportsbook key (e.g., 'pinnacle')", value="pinnacle")

fetch = st.sidebar.button("Fetch live odds")

# -----------------------------
# Fetching & Processing
# -----------------------------
if fetch:
    if not api_key:
        st.error("API key required.")
    else:
        st.info("Fetching live odds...")
        all_rows = []
        try:
            for sport_name in selected_sports:
                # map friendly name to slug; if not found, attempt lowercase keys or use sport_name directly
                sport_slug = sport_options.get(sport_name, sport_name.lower())

                try:
                    raw_events = fetch_odds_theoddsapi(api_key, sport_slug, region=region, market=market_code)
                except requests.HTTPError as e:
                    st.sidebar.error(f"{sport_name}: HTTP error fetching odds: {e}")
                    continue
                except Exception as e:
                    st.sidebar.error(f"{sport_name}: Error fetching odds: {e}")
                    continue

                st.sidebar.write(f"{sport_name}: {len(raw_events)} events fetched")

                for ev in raw_events:
                    evt_name = f"{ev.get('home_team', '')} vs {ev.get('away_team', '')}"
                    commence_time = ev.get("commence_time", "")
                    # collect odds per book for the requested market_code
                    per_book_odds = {}
                    for bmk in ev.get("bookmakers", []):
                        bkey = bmk.get("key")
                        for m in bmk.get("markets", []):
                            if m.get("key") == market_code:
                                # outcomes is list of dicts with 'name' and 'price'
                                outcomes = {out.get("name"): out.get("price") for out in m.get("outcomes", []) if out.get("price") is not None}
                                if outcomes:
                                    per_book_odds[bkey] = outcomes

                    if not per_book_odds:
                        continue

                    # choose sharp book (do NOT mutate user input)
                    chosen_sharp = sharp_book_key_input if sharp_book_key_input in per_book_odds else list(per_book_odds.keys())[0]
                    sharp_odds = per_book_odds.get(chosen_sharp, {})

                    # compute true probabilities from sharp book (remove vig)
                    true_prob_map_raw = {t: american_to_prob(price) for t, price in sharp_odds.items()}
                    true_prob_map = remove_vig_general(true_prob_map_raw)

                    # compute metrics for each book/outcome
                    for bk, od in per_book_odds.items():
                        for selection, price in od.items():
                            try:
                                implied_p = american_to_prob(price)
                                true_p = true_prob_map.get(selection, implied_p)
                                dec = american_to_decimal(price)
                            except Exception:
                                # skip invalid odds
                                continue

                            edge = true_p - implied_p
                            ev_1 = edge * dec
                            roi_pct = (edge / implied_p * 100.0) if implied_p > 0 else 0.0
                            kelly_f = kelly_fraction(true_p, dec)
                            suggested_kelly = (kelly_f * fractional_kelly * bankroll) if use_kelly else np.nan

                            all_rows.append({
                                "sport": sport_name,
                                "event": evt_name,
                                "commence_time": commence_time,
                                "market": market_code,
                                "book": bk,
                                "selection": selection,
                                "american_odds": price,
                                "decimal_odds": round(dec, 3),
                                "implied_prob": round(implied_p, 4),
                                "true_prob": round(true_p, 4),
                                "edge": round(edge, 4),
                                "ev_per_$1": round(ev_1, 4),
                                "roi_%": round(roi_pct, 2),
                                "kelly_frac": round(kelly_f, 4),
                                "suggested_stake_$": round(float(suggested_kelly), 2) if use_kelly else np.nan
                            })

            if not all_rows:
                st.warning("No rows generated — check sport/market/region.")
            else:
                df = pd.DataFrame(all_rows)
                filtered_df = df[(df['ev_per_$1'] >= min_ev) & (df['edge'] >= min_edge) & (df['roi_%'] >= min_roi)]

                st.subheader("Filtered Results")
                columns_to_show = st.multiselect("Columns to display", options=filtered_df.columns, default=filtered_df.columns.tolist())
                st.dataframe(filtered_df[columns_to_show].sort_values(by="edge", ascending=False).reset_index(drop=True), height=700)
                csv = filtered_df.to_csv(index=False)
                st.download_button("Download CSV", data=csv, file_name="filtered_edges.csv", mime="text/csv")

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Configure settings in sidebar and click 'Fetch live odds'.")
