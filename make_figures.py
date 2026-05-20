import pandas as pd
import matplotlib.pyplot as plt

rr = pd.read_csv("data/round_by_round.csv")
res = pd.read_csv("data/results.csv")
na = pd.read_csv("data/n_agent_results.csv")
el = pd.read_csv("data/elimination_results.csv")

elim_configs = {
    "['Q-Learner', 'Q-Learner', 'Greedy', 'Random']": "2 Q-Learners + Greedy + Random",
    "['LLM', 'Q-Learner', 'Greedy', 'Random']": "LLM + Q-Learner + Greedy + Random",
    "['Q-Learner', 'Q-Learner', 'Q-Learner', 'Greedy', 'Greedy']": "3 Q-Learners + 2 Greedy",
}

MONOPOLY = 2.00
COMPETITIVE = 1.00

import os

os.makedirs("figures", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

for matchup in ["Q-Learner vs Q-Learner", "Greedy vs Greedy"]:
    data = rr[rr["matchup"] == matchup]
    by_round = data.groupby("round")["avg_price"].mean()
    smooth = by_round.rolling(20, min_periods=1).mean()
    ax1.plot(smooth.index, smooth.values, label=matchup, lw=1.5)

ax1.axhline(MONOPOLY, color='black', linestyle=':', label='Monopoly ($2.00)')
ax1.axhline(COMPETITIVE, color='gray', linestyle=':', label='Competitive ($1.00)')
ax1.set_xlabel('Round')
ax1.set_ylabel('Average price')
ax1.set_title('(a) Q-Learner and Greedy matchups')
ax1.legend()
ax1.set_ylim(0.9, 2.1)

for matchup in ["LLM vs LLM", "LLM vs Q-Learner", "LLM vs Greedy"]:
    data = rr[rr["matchup"] == matchup]
    by_round = data.groupby("round")["avg_price"].mean()
    smooth = by_round.rolling(10, min_periods=1).mean()
    ax2.plot(smooth.index, smooth.values, label=matchup, lw=1.5)

ax2.axhline(MONOPOLY, color='black', linestyle=':', label='Monopoly ($2.00)')
ax2.axhline(COMPETITIVE, color='gray', linestyle=':', label='Competitive ($1.00)')
ax2.set_xlabel('Round')
ax2.set_ylabel('Average price')
ax2.set_title('(b) LLM matchups')
ax2.legend()
ax2.set_ylim(0.9, 2.1)

plt.tight_layout()
plt.savefig("figures/price_trajectories.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

order = res.sort_values("collusion_index", ascending=False)

plt.figure(figsize=(10, 4.5))
x_pos = range(len(order))

bar_colors = []
for m in order["matchup"]:
    if "LLM" in m:
        bar_colors.append("darkred")
    else:
        bar_colors.append("steelblue")

plt.bar(x_pos, order["collusion_index"],
        color=bar_colors, alpha=0.8, label="Overall")
plt.bar(x_pos, order["late_game_collusion_index"],
        color=bar_colors, alpha=0.4, label="Late game",
        edgecolor=bar_colors, linewidth=1)

plt.xticks(x_pos, order["matchup"], rotation=30, ha="right")
plt.ylabel("Collusion index (0=competitive, 1=monopoly)")
plt.title("Collusion index by matchup")
plt.ylim(0, 0.75)
plt.legend()
plt.tight_layout()
plt.savefig("figures/collusion_index.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

na_grouped = na.groupby(["config", "round"])["avg_market_price"].mean()

config_names = {
    "['Q-Learner', 'Q-Learner', 'Q-Learner']": "3 Q-Learners",
    "['Q-Learner', 'Q-Learner', 'Q-Learner', 'Q-Learner', 'Q-Learner']": "5 Q-Learners",
    "['Q-Learner', 'Q-Learner', 'Greedy']": "2 Q-Learners + Greedy",
    "['LLM', 'Q-Learner', 'Q-Learner']": "LLM + 2 Q-Learners",
    "['LLM', 'LLM', 'Q-Learner']": "2 LLMs + Q-Learner",
    "['LLM', 'Q-Learner', 'Greedy']": "LLM + Q-Learner + Greedy",
}


plt.figure(figsize=(10, 4.5))

colors = plt.cm.Set2(range(len(config_names)))

ax = plt.gca()

for i, (cfg, label) in enumerate(config_names.items()):

    prices = na_grouped.loc[cfg]
    smooth = prices.rolling(30, min_periods=1).mean()

    linestyle = '-' if "LLM" in cfg else '--'

    jitter = i * 0.01
    y = smooth.values + jitter

    ax.plot(
        smooth.index,
        y,
        label=label,
        color=colors[i],
        lw=1.5,
        linestyle=linestyle
    )

ax.axhline(MONOPOLY, color='black', linestyle=':', alpha=0.7)
ax.axhline(COMPETITIVE, color='gray', linestyle=':', alpha=0.7)

ax.set_xlabel("Round")
ax.set_ylabel("Average price")
ax.set_title("N-agent markets over 500 rounds")

ax.set_ylim(0.9, 2.1)
ax.set_xlim(0, 500)

ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("figures/n_agent.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

markers = ['o', 's', '^']
colors = ['blue', 'orange', 'green']
linestyles = ['-', '--', ':']

for i, (cfg, label) in enumerate(elim_configs.items()):
    data = el[el["config"] == cfg]

    active = data.groupby("round")["n_active"].mean().sort_index()

    jitter = i * 0.05
    y = active.values + jitter

    ax1.plot(active.index, y, label=label, color=colors[i], linestyle=linestyles[i], marker=markers[i],
             markevery=50, lw=2.5, zorder=10 - i, alpha=0.9)

    prices = data.groupby("round")["avg_price"].mean().sort_index()
    smooth = prices.rolling(10, min_periods=1).mean()

    ax2.plot(smooth.index, smooth.values, label=label, color=colors[i], lw=2, linestyle=linestyles[i], zorder=10 - i)

ax1.set_xlabel("Round")
ax1.set_ylabel("Active agents")
ax1.set_title("(a) Agents surviving")
ax1.legend(fontsize=9)

ax1.set_ylim(0, 5.5)
ax1.set_xlim(0, 150)

plt.tight_layout()
plt.savefig("figures/elimination.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("All figures saved to figures/")