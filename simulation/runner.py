from agents.base import Agent
from simulation.market import BertrandMarket
from config import NUM_ROUNDS, LLM_ROUNDS


class RoundRecord:
    def __init__(self, run, round_num, agent_a, agent_b, price_a, price_b, profit_a, profit_b):
        self.run = run
        self.round_num = round_num
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.price_a = price_a
        self.price_b = price_b
        self.profit_a = profit_a
        self.profit_b = profit_b


class MatchupResult:
    def __init__(self, agent_a, agent_b):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.rounds = []

    def avg_price_a(self):
        total = 0
        for r in self.rounds:
            total += r.price_a

        return total / len(self.rounds)

    def avg_price_b(self):
        total = 0
        for r in self.rounds:
            total += r.price_b

        return total / len(self.rounds)

    def avg_profit_a(self):
        total = 0
        for r in self.rounds:
            total += r.profit_a

        return total / len(self.rounds)

    def avg_profit_b(self):
        total = 0
        for r in self.rounds:
            total += r.profit_b

        return total / len(self.rounds)


def run_game(agent_a, agent_b, market, num_rounds, run_id):
    history_a = []
    history_b = []
    records = []

    for r in range(num_rounds):
        price_a = agent_a.choose_price(market.price_grid, history_a)
        price_b = agent_b.choose_price(market.price_grid, history_b)

        profit_a, profit_b = market.step([price_a, price_b])

        agent_a.update(profit_a)
        agent_b.update(profit_b)

        record = RoundRecord(
            run=run_id,
            round_num=r + 1,
            agent_a=agent_a.name,
            agent_b=agent_b.name,
            price_a=price_a,
            price_b=price_b,
            profit_a=profit_a,
            profit_b=profit_b,
        )
        records.append(record)

        history_a.append({
            "round": r + 1,
            "my_price": price_a,
            "opp_price": price_b,
            "my_profit": profit_a,
            "opp_profit": profit_b,
        })
        history_b.append({
            "round": r + 1,
            "my_price": price_b,
            "opp_price": price_a,
            "my_profit": profit_b,
            "opp_profit": profit_a,
        })

        print(f"Round {r + 1}: {agent_a.name} ${price_a:.3f}, {agent_b.name} ${price_b:.3f}, "
              f"profits ({profit_a:.4f}, {profit_b:.4f})")


    return records


def run_matchup(agent_a_cls, agent_b_cls, market, num_runs, is_llm_game):
    num_rounds = LLM_ROUNDS if is_llm_game else NUM_ROUNDS

    tmp_a = agent_a_cls()
    tmp_b = agent_b_cls()
    matchup_name = f"{tmp_a.name} vs {tmp_b.name}"
    result = MatchupResult(agent_a=tmp_a.name, agent_b=tmp_b.name)

    print(f"Matchup: {matchup_name}  ({num_runs} runs * {num_rounds} rounds)")

    for run in range(num_runs):
        agent_a = agent_a_cls()
        agent_b = agent_b_cls()

        print(f"Run {run + 1}/{num_runs}")

        records = run_game(agent_a, agent_b, market, num_rounds, run)
        result.rounds.extend(records)

    print(f"Avg price: {result.avg_price_a()}, {result.avg_price_b()}")
    print(f"Avg profit: {result.avg_profit_a()}, {result.avg_profit_b()}")

    return result


def run_n_agent_game(agents, market, num_rounds, run_id):
    n = len(agents)
    histories = [[] for _ in range(n)]
    records = []

    for r in range(num_rounds):
        prices = [agent.choose_price(market.price_grid, histories[i]) for i, agent in enumerate(agents)]
        profits = market.step(prices)

        for i, agent in enumerate(agents):
            opp_prices = [prices[j] for j in range(n) if j!=i]
            avg_opp_price = sum(opp_prices) / len(opp_prices)
            agent.update(profits[i])

            histories[i].append({
                "round": r + 1,
                "my_price": prices[i],
                "opp_price": avg_opp_price,
                "all_prices": prices,
                "my_profit": profits[i],
                "opp_profit": sum(profits) - profits[i],
            })

        avg_price = sum(prices) / n
        print(f"Round {r+1}: prices={[f'{p:.2}' for p in prices]} "
              f"profits={[f'{p:.2f}' for p in profits]} avg={avg_price:.2f}")

        records.append({
            "run": run_id,
            "round": r + 1,
            "n_agents": n,
            "prices": prices,
            "profits": profits,
            "avg_price": avg_price,
            "agent_names": [a.name for a in agents],
        })

    return records


def run_n_agent_matchup(agent_classes, market, num_runs, num_rounds):
    names = [cls().name for cls in agent_classes]
    print(f"\nN-Agent matchup: {names} ({num_runs} runs * {num_rounds} rounds)")
    all_records = []
    for run in range(num_runs):
        agents = [cls() for cls in agent_classes]
        records = run_n_agent_game(agents, market, num_rounds, run)
        all_records.extend(records)
    return all_records
