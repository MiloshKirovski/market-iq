import random
from agents.base import Agent
from config import GREEDY_UNDERCUT

class GreedyAgent(Agent):
    # Greedy undercutter: always tries to price just below the opponent.
    # Simple myopic business strategy
    def __init__(self, undercut_margin=GREEDY_UNDERCUT):
        super().__init__(name="Greedy")
        self.undercut_margin = undercut_margin

    def choose_price(self, price_grid, history):
        if not history:
            price = max(price_grid)
        else:
            last_opp_price = history[-1]["opp_price"]
            target = last_opp_price - self.undercut_margin

            valid = [p for p in price_grid if p <= target]
            if valid:
                price = max(valid)
            else:
                price = min(price_grid)

        self.last_price = price
        return price

    def update(self, my_profit):
        pass


