import os

from simulation.market import BertrandMarket
from simulation.runner import run_matchup
from analysis.metrics import records_to_dataframe, compute_benchmark_table, late_game_collusion

from agents import RandomAgent, GreedyAgent, QLearningAgent, LLMAgent

from config import NUM_RUNS, DATA_DIR

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    market = BertrandMarket()

    matchups = [
        (QLearningAgent, QLearningAgent, False),
        (QLearningAgent, GreedyAgent, False),
        (QLearningAgent, RandomAgent, False),

        (GreedyAgent, GreedyAgent, False),
        (RandomAgent, RandomAgent, False),

        (RandomAgent, RandomAgent, False),

        (LLMAgent, LLMAgent, True),
        (LLMAgent, QLearningAgent, True),
        (LLMAgent, GreedyAgent, True),
        (LLMAgent, RandomAgent, True),
    ]

    all_results = []

    for agent_a, agent_b, is_llm in matchups:
        result = run_matchup(agent_a, agent_b, market, num_runs=NUM_RUNS, is_llm_game=is_llm)
        all_results.append(result)

    df_rounds = records_to_dataframe(all_results)

    df_summary = compute_benchmark_table(df_rounds, market)

    df_late = late_game_collusion(df_rounds, market)

    df_summary = df_summary.merge(df_late, on="matchup", how="left")

    df_rounds.to_csv(f"{DATA_DIR}/round_by_round.csv", index=False)
    df_summary.to_csv(f"{DATA_DIR}/results.csv", index=False)

    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()