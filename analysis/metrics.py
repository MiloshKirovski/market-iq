import pandas as pd
import numpy as np
from simulation.market import BertrandMarket
from simulation.runner import MatchupResult


def records_to_dataframe(results):
    rows = []
    for result in results:
        for r in result.rounds:
            rows.append({
                "run": r.run,
                "round": r.round_num,
                "agent_a": r.agent_a,
                "agent_b": r.agent_b,
                "matchup": f"{r.agent_a} vs {r.agent_b}",
                "price_a": r.price_a,
                "price_b": r.price_b,
                "profit_a": r.profit_a,
                "profit_b": r.profit_b,
                "avg_price": (r.price_a + r.price_b) / 2,
            })

    return pd.DataFrame(rows)


def compute_benchmark_table(df, market):
    records = []

    for matchup, group in df.groupby("matchup"):
        agent_a = group["agent_a"].iloc[0]
        agent_b = group["agent_b"].iloc[0]

        avg_price_a = group["price_a"].mean()
        avg_price_b = group["price_b"].mean()
        avg_profit_a = group["profit_a"].mean()
        avg_profit_b = group["profit_b"].mean()
        avg_price = group["avg_price"].mean()

        ci = market.collusion_index(avg_price)

        var_a = group["profit_a"].std()
        var_b = group["profit_b"].std()

        conv_round = _estimate_convergence(group["avg_price"].values)

        records.append({
            "matchup": matchup,
            "agent_a": agent_a,
            "agent_b": agent_b,
            "avg_price_a": round(avg_price_a, 4),
            "avg_price_b": round(avg_price_b, 4),
            "avg_profit_a": round(avg_profit_a, 4),
            "avg_profit_b": round(avg_profit_b, 4),
            "collusion_index": round(ci, 4),
            "price_var_a": round(var_a, 4),
            "price_var_b": round(var_b, 4),
            "convergence_round": conv_round,
        })

    return pd.DataFrame(records).sort_values("collusion_index", ascending=False)


def _estimate_convergence(prices, window=20, threshold=0.02):
    if len(prices) < window:
        return -1
    for i in range(window, len(prices)):
        if np.std(prices[i - window:i]) < threshold:
            return i
    return -1


def late_game_collusion(df, market, last_n=200):
    rows = []
    for matchup, group in df.groupby("matchup"):
        last = group.nlargest(last_n, "round")
        ci = market.collusion_index(last["avg_price"].mean())
        rows.append({"matchup": matchup, "late_game_collusion_index": round(ci, 4)})
    return pd.DataFrame(rows)
