import random
from agents.base import Agent

class RandomAgent(Agent):
    def __init__(self, seed):
        super().__init__(name="Random")
        self.rng = random.Random(seed)

    def choose_action(self, price_grid, history):
        price = self.rng.choice(price_grid)
        self.last_price = price
        return price

    def update(self, my_price, opp_price, my_profit, opp_profit):
        pass
        