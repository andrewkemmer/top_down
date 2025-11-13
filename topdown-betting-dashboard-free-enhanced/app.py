import os
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Sharp EV Top-Down Betting Dashboard")

# ---- Helper functions ----
def american_to_decimal(odds):
    odds = float(odds)
    return 1.0 + (odds/100.0) if odds > 0 else 1.0 + (100.0 / -odds)

def american_to_prob(odds):
    odds = float(odds)
    return 100.0/(odds+100.0) if odds>0 else -odds/(-odds+100.0)

def remove_vig_general(probs):
    s = sum(probs)
    return [p/s for p in probs] if s>0 else probs

def kelly_fraction(true_p, decimal_odds):
    b = decimal_odds-1
    f = (b*true_p-(1-true_p))/b
    return max(f,0.0)

THEODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_odds_theoddsapi(api_key, sport, region="us", market="h2h"):
    endpoint = f"{THEODDS_BASE}/sports/{sport}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": market, "oddsFormat": "american"}
    r = requests.get(endpoint, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ---- Streamlit UI ----
st.title("🔎 Sharp EV Top-Down Betting Dashboard")

st.sidebar.header("API / Data Source")
api_key = st.sidebar.text_input("TheOddsAPI Key", value=os.getenv("THEODDS_API_KEY", ""))

# Dynamic sport selection
if api_key:
    try:
        sports_list = requests.get(f"{THEODDS_BASE}/sports/?apiKey={api_key}").json()
        sport_options = {sport['title']: sport['key'] for sport in sports_list}
        selected_sport = st.sidebar.selectbox("Select sport", list(sport_options.keys()))
        sport_slug = sport_options[selected_sport]
    except Exception as e:
        st.sidebar.error(f"Error fetching sports: {e}")
        sport_slug = "basketball_nba"
else:
    sport_slug = "basketball_nba"

st.sidebar.header("Market")
market_type = st.sidebar.selectbox("Market type", ["h2h (moneyline)", "spreads", "totals"])
region = st.sidebar.selectbox("Region", ["us", "uk", "eu", "au"], index=0)

st.sidebar.header("Bet Sizing")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1.0, value=1000.0, step=1.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
use_kelly = st.sidebar.checkbox("Show Kelly stake", value=True)

st.sidebar.header("Filters")
min_ev = st.sidebar.number_input("Min EV per $1", value=0.0)
min_edge = st.sidebar.number_input("Min edge", value=0.0)
min_roi = st.sidebar.number_input("Min ROI %", value=0.0)

st.sidebar.header("Sharp Book Settings")
sharp_book_key = st.sidebar.text_input("Sharp sportsbook key (e.g., 'pinnacle')", value="pinnacle")

fetch = st.sidebar.button("Fetch live odds")

if fetch:
    if not api_key:
        st.error("API key required.")
    else:
        st.info("Fetching live odds...")
        try:
            market_code = "h2h" if market_type.startswith("h2h") else ("spreads" if market_type=="spreads" else "totals")
            raw_events = fetch_odds_theoddsapi(api_key, sport_slug, region=region, market=market_code)

            st.sidebar.subheader("Debug / Events Info")
            st.sidebar.write(f"Events fetched: {len(raw_events)}")

            all_rows = []
            for ev in raw_events:
                evt_name = ev.get("home_team", "") + " vs " + ev.get("away_team", "")
                commence_time = ev.get("commence_time", "")
                teams = ev.get("teams", [])
                per_book_odds = {}
                for bmk in ev.get("bookmakers", []):
                    bkey = bmk.get("key")
                    for m in bmk.get("markets", []):
                        if m.get("key") == market_code:
                            outcomes = m.get("outcomes", [])
                            od = {out.get("name"):out.get("price") for out in outcomes if out.get("price") is not None}
                            if od:
                                per_book_odds[bkey] = od
                if not per_book_odds:
                    continue

                # ---- Compute true probability based on sharp book ----
                sharp_odds = per_book_odds.get(sharp_book_key, {})
                if not sharp_odds:
                    # fallback: use first available book
                    sharp_book_key = list(per_book_odds.keys())[0]
                    sharp_odds = per_book_odds[sharp_book_key]

                true_prob_map = {t: american_to_prob(sharp_odds[t]) for t in sharp_odds.keys()}
                true_prob_map = dict(zip(true_prob_map.keys(), remove_vig_general(list(true_prob_map.values()))))

                # ---- Compute edges for all books ----
                for bk, od in per_book_odds.items():
                    for team in od.keys():
                        implied_p = american_to_prob(od[team])
                        true_p = true_prob_map[team]  # baseline from sharp book
                        edge = true_p - implied_p
                        dec = american_to_decimal(od[team])
                        ev_1 = edge * dec
                        roi_pct = (edge / implied_p * 100.0) if implied_p > 0 else 0.0
                        kelly_f = kelly_fraction(true_p, dec)
                        suggested_kelly = kelly_f * fractional_kelly * bankroll if use_kelly else np.nan

                        all_rows.append({
                            "event": evt_name,
                            "commence_time": commence_time,
                            "market": market_code,
                            "book": bk,
                            "selection": team,
                            "american_odds": od[team],
                            "decimal_odds": round(dec,3),
                            "implied_prob": round(implied_p,4),
                            "true_prob": round(true_p,4),
                            "edge": round(edge,4),
                            "ev_per_$1": round(ev_1,4),
                            "roi_%": round(roi_pct,2),
                            "kelly_frac": round(kelly_f,4),
                            "suggested_stake_$": round(float(suggested_kelly),2) if use_kelly else np.nan
                        })

            if not all_rows:
                st.warning("No rows generated — check sport/market/region.")
            else:
                df = pd.DataFrame(all_rows)
                filtered_df = df[(df['ev_per_$1']>=min_ev) & (df['edge']>=min_edge) & (df['roi_%']>=min_roi)]
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
