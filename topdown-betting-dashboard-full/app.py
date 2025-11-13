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

st.title("🔎 Top-Down Betting Dashboard")
st.sidebar.header("API / Data Source")
api_key = st.sidebar.text_input("TheOddsAPI Key (or set THEODDS_API_KEY env var)", value=os.getenv("THEODDS_API_KEY", ""))
st.sidebar.header("Market / Sport")
sport = st.sidebar.text_input("Sport slug (TheOddsAPI)", value="basketball_nba")
market_type = st.sidebar.selectbox("Market type", ["h2h (moneyline)", "spreads", "totals"])
region = st.sidebar.selectbox("Region", ["us", "uk", "eu", "au"], index=0)
st.sidebar.header("Book Weights")
weights_text = st.sidebar.text_area("JSON style: {'book_key': weight, ...}", height=120, value="")
st.sidebar.header("Bet Sizing")
use_kelly = st.sidebar.checkbox("Show Kelly stake", value=True)
fractional_kelly = st.sidebar.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
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
            if not raw_events:
                st.warning("No events returned. Try different sport/region.")
            else:
                all_rows = []
                books_seen = set()
                for ev in raw_events:
                    for bmk in ev.get("bookmakers", []):
                        books_seen.add(bmk.get("key"))
                st.sidebar.subheader("Books detected")
                st.sidebar.write(", ".join(sorted(books_seen)))
                user_weights = {}
                if weights_text.strip():
                    try:
                        user_weights = eval(weights_text.strip(), {}, {})
                    except:
                        st.sidebar.error("Couldn't parse weights.")
                weights = user_weights if user_weights else {b:1.0/len(books_seen) for b in books_seen}
                total_w = sum(weights.values())
                for k in list(weights.keys()):
                    weights[k] = float(weights[k]) / float(total_w)
                st.sidebar.subheader("Normalized weights")
                st.sidebar.write(weights)

                for ev in raw_events:
                    evt_name = ev.get("home_team") + " vs " + ev.get("away_team")
                    commence_time = ev.get("commence_time", "")
                    if market_code == "h2h":
                        per_book_odds = {}
                        teams = ev.get("teams", [])
                        for bmk in ev.get("bookmakers", []):
                            bkey = bmk.get("key")
                            for m in bmk.get("markets", []):
                                if m.get("key") == "h2h":
                                    outcomes = m.get("outcomes", [])
                                    od = {out.get("name"):out.get("price") for out in outcomes}
                                    if od:
                                        per_book_odds[bkey] = od
                        valid_books = {bk:od for bk,od in per_book_odds.items() if all(t in od for t in teams)}
                        if not valid_books:
                            continue
                        team_probs_weighted = {t: 0.0 for t in teams}
                        for bk,od in valid_books.items():
                            w = weights.get(bk,0.0)
                            for team in team_probs_weighted.keys():
                                team_probs_weighted[team] += w * american_to_prob(od[team])
                        true_probs = remove_vig_general([team_probs_weighted[t] for t in team_probs_weighted])
                        true_prob_map = dict(zip(team_probs_weighted.keys(), true_probs))
                        for bk,od in valid_books.items():
                            for team in team_probs_weighted.keys():
                                implied_p = american_to_prob(od[team])
                                true_p = true_prob_map[team]
                                edge = true_p - implied_p
                                dec = american_to_decimal(od[team])
                                payout = dec
                                ev_1 = edge * payout
                                roi_pct = (edge / implied_p * 100.0) if implied_p>0 else 0.0
                                kelly_f = kelly_fraction(true_p, dec)
                                suggested_kelly = kelly_f * fractional_kelly * bankroll if use_kelly else np.nan
                                all_rows.append({
                                    "event": evt_name,
                                    "commence_time": commence_time,
                                    "market": "moneyline",
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
                    st.warning("No rows generated. Try another sport/market.")
                else:
                    df = pd.DataFrame(all_rows)
                    df_sorted = df.sort_values(by="edge", ascending=False).reset_index(drop=True)
                    st.subheader("Results — top edges")
                    st.dataframe(df_sorted.head(200), height=700)
                    st.subheader("Summary")
                    total_ev = df_sorted['ev_per_$1'].sum()
                    avg_edge = df_sorted['edge'].mean()
                    positive_count = (df_sorted['edge']>0).sum()
                    st.markdown(f"- Rows analyzed: **{len(df_sorted)}**")
                    st.markdown(f"- Positive-edge rows: **{int(positive_count)}**")
                    st.markdown(f"- Average edge: **{avg_edge:.4f}**")
                    st.markdown(f"- Sum EV per $1 if taking every positive edge once: **{total_ev:.4f}**")
                    csv = df_sorted.to_csv(index=False)
                    st.download_button("Download CSV (full results)", data=csv, file_name="true_line_edges.csv", mime="text/csv")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Configure settings in sidebar and click 'Fetch live odds'.")
