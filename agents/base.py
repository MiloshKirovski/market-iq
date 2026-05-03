class Agent:
    def __init__(self, name):
        self.name = name
        self.last_price = None

    def choose_price(self, price_grid, history):
        # price_grid - list of valid discrete price levels
        # history - list of dicts with past round data
        raise NotImplementedError(f"{self.__class__.__name__} must implement choose_price()")

    def update(self, my_price, opp_price, my_profit, opp_profit):
        # Updating internal state after observing round outcome (after both agents chose price)
        raise NotImplementedError(f"{self.__class__.__name__} must implement update()")

    def reset(self):
        self.last_price = None

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

