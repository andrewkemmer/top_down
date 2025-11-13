
import os
import math
import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Top-Down Betting Dashboard")

def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / -odds)

def american_to_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)

def remove_vig_general(probs):
    s = sum(probs)
    if s <= 0:
        return probs
    return [p / s for p in probs]

def kelly_fraction(true_p, decimal_odds):
    if decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - true_p
    f = (b * true_p - q) / b
    return max(f, 0.0)

THEODDS_BASE = "https://api.the-odds-api.com/v4"

def fetch_odds_theoddsapi(api_key, sport, region="us", market="h2h"):
    endpoint = f"{THEODDS_BASE}/sports/{sport}/odds"
    params = {"apiKey": api_key, "regions": region, "markets": market, "oddsFormat": "american"}
    r = requests.get(endpoint, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

st.title("🔎 Top-Down Betting Dashboard (Free API compatible)")

st.sidebar.header("API / Data Source")
api_key = st.sidebar.text_input("TheOddsAPI Key", value=os.getenv("THEODDS_API_KEY", ""))
st.sidebar.header("Sport / Market")
sport = st.sidebar.text_input("Sport slug (TheOddsAPI)", value="basketball_nba")
market_type = st.sidebar.selectbox("Market type", ["h2h (moneyline)", "spreads", "totals"])
region = st.sidebar.selectbox("Region", ["us", "uk", "eu", "au"], index=0)
st.sidebar.header("Bet Sizing")
use_kelly = st.sidebar.checkbox("Show Kelly stake", value=True)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1.0, value=1000.0, step=1.0)
fetch = st.sidebar.button("Fetch live odds")

if fetch:
    if not api_key:
        st.error("API key required.")
    else:
        st.info("Fetching live odds...")
        try:
            market_code = "h2h" if market_type.startswith("h2h") else ("spreads" if market_type=="spreads" else "totals")
            raw_events = fetch_odds_theoddsapi(api_key, sport, region=region, market=market_code)
            st.sidebar.subheader("Debug / Events Info")
            st.sidebar.write(f"Events fetched: {len(raw_events)}")
            if not raw_events:
                st.warning("No events returned. Try different sport/region.")
            else:
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
                    team_probs_weighted = {t: 0.0 for t in teams}
                    weights = {b:1.0/len(per_book_odds) for b in per_book_odds.keys()}
                    for bk,od in per_book_odds.items():
                        w = weights.get(bk,0.0)
                        for team in team_probs_weighted.keys():
                            if team in od:
                                team_probs_weighted[team] += w * american_to_prob(od[team])
                    true_probs = remove_vig_general([team_probs_weighted[t] for t in team_probs_weighted])
                    true_prob_map = dict(zip(team_probs_weighted.keys(), true_probs))
                    for bk,od in per_book_odds.items():
                        for team in od.keys():
                            implied_p = american_to_prob(od[team])
                            true_p = true_prob_map.get(team, implied_p)
                            edge = true_p - implied_p
                            dec = american_to_decimal(od[team])
                            ev_1 = edge * dec
                            roi_pct = (edge / implied_p * 100.0) if implied_p>0 else 0.0
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
                    df_sorted = df.sort_values(by="edge", ascending=False).reset_index(drop=True)
                    st.subheader("Results — top edges")
                    st.dataframe(df_sorted.head(200), height=700)
                    st.subheader("Summary")
                    st.markdown(f"- Rows analyzed: **{len(df_sorted)}**")
                    positive_count = (df_sorted['edge']>0).sum()
                    st.markdown(f"- Positive-edge rows: **{int(positive_count)}**")
                    st.markdown(f"- Average edge: **{df_sorted['edge'].mean():.4f}**")
                    st.markdown(f"- Sum EV per $1 if taking every positive edge once: **{df_sorted['ev_per_$1'].sum():.4f}**")
                    csv = df_sorted.to_csv(index=False)
                    st.download_button("Download CSV (full results)", data=csv, file_name="true_line_edges.csv", mime="text/csv")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Configure settings in sidebar and click 'Fetch live odds'.")
