import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SupeRR Selector S2 – Pressure Ratings", layout="wide")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data") + os.sep


@st.cache_data
def load():
    d = dict(
        bat=pd.read_csv(D + "batters_pressure_ratings.csv"), bow=pd.read_csv(D + "bowlers_pressure_ratings.csv"),
        tb=pd.read_csv(D + "final_top10_batters.csv"), tw=pd.read_csv(D + "final_top10_bowlers.csv"),
        li=pd.read_csv(D + "li_by_over.csv"), cmp=pd.read_csv(D + "pressure_vs_normal.csv"),
        mb=pd.read_csv(D + "batter_pressure_matches.csv"), mw=pd.read_csv(D + "bowler_pressure_matches.csv"),
    )
    for k, f in [("rank", "rank_stability.csv"), ("ex", "example_chase.csv")]:   # optional extra files
        d[k] = pd.read_csv(D + f) if os.path.exists(D + f) else None
    return d


d = load()
st.title("Most Reliable Players Under Pressure")
st.caption("IPL + T20I 2022–26 · pressure = top-25% leverage deliveries (win-probability swing) · ratings shrunk toward each player's own baseline")
t1, t2, t3, t4 = st.tabs(["Top 10", "Pressure definition", "Player explorer", "Method"])

# ------------------------------------------------------------------ TOP 10
with t1:
    a, b = st.columns(2)
    a.subheader("Top 10 batters")
    a.dataframe(d["tb"][["rank", "name", "score", "P_shr", "cons", "p_balls", "nm", "p_SR", "p_top10"]].round(2), hide_index=True, use_container_width=True)
    b.subheader("Top 10 bowlers")
    b.dataframe(d["tw"][["rank", "name", "score", "P_shr", "cons", "p_balls_r", "nm", "p_econ", "p_dot%", "p_top10"]].round(2), hide_index=True, use_container_width=True)
    st.caption("rank = median rank over 100 match-level bootstrap resamples (bagging). score = 0.5·pressure level + 0.25·lift vs own baseline + 0.25·consistency. "
               "p_top10 = share of resamples in which the player stays in the top 10.")

    c1, c2 = st.columns(2)
    c1.markdown("**Composite score – batters**")
    c1.bar_chart(d["tb"].set_index("name")["score"], horizontal=True, color="#c2185b")
    c2.markdown("**Composite score – bowlers**")
    c2.bar_chart(d["tw"].set_index("name")["score"], horizontal=True, color="#1565c0")

    st.subheader("Does pressure change performance? (usual level vs pressure level)")
    side0 = st.radio("Players", ["Batters", "Bowlers"], horizontal=True, key="sc")
    df0, top0 = (d["bat"], d["tb"]) if side0 == "Batters" else (d["bow"], d["tw"])
    sc = df0[df0.eligible].copy()
    sc["group"] = np.where(sc.name.isin(top0.name), "Top 10", "Other eligible")
    sc = sc.rename(columns={"b_z": "Usual level (ordinary balls)", "p_z": "Pressure level (raw)"})
    st.scatter_chart(sc, x="Usual level (ordinary balls)", y="Pressure level (raw)", color="group", size=60)
    st.caption("Points above the diagonal rise under pressure; below it they drop. Most players sit close to the line, so pressure performance mostly reflects overall quality.")

    if d["rank"] is not None:
        st.subheader("Rank stability (100 bootstrap re-samples of matches)")
        rs = d["rank"]
        rc1, rc2 = st.columns(2)
        for col, role in [(rc1, "Batter"), (rc2, "Bowler")]:
            x = rs[rs.role == role].sort_values("rank_median").head(15).set_index("name")
            col.markdown(f"**{role}s – chance of staying in the top 10**")
            col.bar_chart(x["p_top10"], horizontal=True, color="#2e7d32")
            col.dataframe(x[["rank_p25", "rank_median", "rank_p75"]].rename(columns={"rank_p25": "best-case rank (25%)", "rank_median": "median rank", "rank_p75": "worst-case rank (75%)"}), use_container_width=True)

# ------------------------------------------------------------------ PRESSURE DEFINITION
with t2:
    st.subheader("Leverage by over (average LI; 1.0 = average delivery)")
    li = d["li"].pivot(index="overid", columns="innings", values="LI"); li.columns = ["1st innings", "2nd innings"]
    st.line_chart(li)
    st.subheader("Share of deliveries classed as pressure, by over")
    sh = d["li"].pivot(index="overid", columns="innings", values="pressure_share"); sh.columns = ["1st innings", "2nd innings"]
    st.bar_chart(sh)

    st.subheader("Pressure vs ordinary deliveries")
    c = d["cmp"].copy(); c["type"] = c.pressure.map({True: "Pressure", False: "Ordinary"}); c = c.set_index("type")
    k1, k2, k3 = st.columns(3)
    k1.markdown("**Average leverage**"); k1.bar_chart(c["avg_LI"])
    k2.markdown("**Run rate (per over)**"); k2.bar_chart(c["run_rate"])
    k3.markdown("**Closeness of match (|WP − 0.5|, lower = tighter)**"); k3.bar_chart(c["avg_wp_before"])

    if d["ex"] is not None:
        ex = d["ex"].reset_index(drop=True)
        st.subheader(f"Example chase: {ex.home_team.iat[0]} v {ex.away_team.iat[0]} ({ex.match_date.iat[0]})")
        w = pd.DataFrame({"Ball": np.arange(1, len(ex) + 1), "Win probability (chasing side)": ex.wp_before,
                          "Pressure balls": np.where(ex.pressure, ex.wp_before, np.nan)}).set_index("Ball")
        st.line_chart(w)
        st.caption("Win probability of the chasing side ball by ball; the second line marks deliveries our model classes as pressure.")
        wk = ex[ex.is_wicket][["overid", "over_ball_num", "striker_name", "bowler_name", "wp_before"]].rename(columns={"wp_before": "WP before wicket"})
        st.markdown("**Wickets in this chase**"); st.dataframe(wk.round(2), hide_index=True)

# ------------------------------------------------------------------ PLAYER EXPLORER
with t3:
    side = st.radio("View", ["Batters", "Bowlers"], horizontal=True)
    df, mm, pc = (d["bat"], d["mb"], "striker_name") if side == "Batters" else (d["bow"], d["mw"], "bowler_name")
    df = df[df.eligible]
    names = list(df.sort_values("rank").name)
    name = st.selectbox("Player", names)
    r = df[df.name == name].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rank", int(r["rank"])); k2.metric("Score", f"{r.score:.2f}"); k3.metric("Consistency", f"{r.cons:.0%}"); k4.metric("P(top 10)", f"{r.p_top10:.0%}")

    x = mm[mm[pc] == name].sort_values("match_date").reset_index(drop=True)
    x.index = x.index + 1; x.index.name = "Pressure match #"
    x["Cumulative impact"] = x.z.cumsum()
    g1, g2 = st.columns(2)
    g1.markdown("**Cumulative pressure impact**"); g1.line_chart(x["Cumulative impact"])
    g2.markdown("**Impact in each match** (green = helped team, red = hurt)")
    g2.bar_chart(pd.DataFrame({"Positive": x.z.clip(lower=0), "Negative": x.z.clip(upper=0)}), color=["#2e7d32", "#c62828"])
    st.dataframe(x.drop(columns=["match_id"]).round(2))

    st.subheader("Compare two players")
    cc1, cc2 = st.columns(2)
    pa = cc1.selectbox("Player A", names, index=0, key="pa"); pb = cc2.selectbox("Player B", names, index=min(1, len(names) - 1), key="pb")
    cols = {"P_shr": "Pressure level", "delta": "Lift vs usual", "cons": "Consistency"}
    cmp_df = df[df.name.isin([pa, pb])].set_index("name")[list(cols)].rename(columns=cols).T
    st.bar_chart(cmp_df)

# ------------------------------------------------------------------ METHOD
with t4:
    st.markdown("""
1. **Win-probability model** per innings (monotone gradient boosting, match-grouped out-of-fold predictions).
2. **Leverage** = spread of next-ball WP across league outcomes → pressure = top 25% of deliveries.
3. **Delivery z-score** = (WP after actual outcome − expected WP) / leverage → comparable between pressure and ordinary balls.
4. **Pooling**: pressure score shrunk toward the player's own baseline; **consistency** = share of pressure matches with positive net z.
5. **Eligibility**: ≥10 pressure matches and ≥90 pressure balls faced / ≥72 bowled; matches with non-Test-playing sides excluded from ratings.
6. **Final rank** = median rank over 100 match-level bootstrap resamples (bagging).
7. **Reliability**: bootstrap rank stability, split-half and 2025–26 hold-out checks (see report).
""")