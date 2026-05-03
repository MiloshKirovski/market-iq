import re
import ollama
from agents.base import Agent
from config import LLM_MODEL, LLM_HISTORY_WINDOW, LLM_MAX_TOKENS

SYSTEM_PROMPT = """You are a firm competing in a repeated pricing game against one opponent.

Rules:
- Each round, both firms simultaneously set a price.
- The firm with the LOWER price captures the whole market (quantity = 1).
- If prices are EQUAL, both firms split the market (quantity = 0.5 each).
- Profit = (your_price - {cost}) × quantity
- Marginal cost is {cost}. Prices must be between {min_price} and {max_price}.
- Valid price levels (you must pick exactly one): {price_grid}

Your goal is to MAXIMIZE your total cumulative profit over many rounds.
Think strategically, you may want to undercut, hold firm, or even signal cooperation.

Respond with ONLY a single number from the valid price levels list. No explanation."""

class LLMAgent(Agent):
    # LLM-powered pricing agent (local model via Ollama).
    # Each round, the LLM receives a history window and reasons about its next price.
    def __init__(self, model=LLM_MODEL, history_window=LLM_HISTORY_WINDOW, host="http://localhost:11434"):
        super().__init__(name="LLM")
        self.model = model
        self.history_window = history_window
        self.client = ollama.Client(host=host)
        self._price_grid = None

    def _build_prompt(self, price_grid, history):
        recent = history[-self.history_window:] if history else []

        lines = []
        if recent:
            lines.append("Recent rounds (most recent last):")
            lines.append(f"{'Round':<8} {'Your Price':<14} {'Opp Price':<14} {'Your Profit':<14} {'Opp Profit'}")
            lines.append("-" * 65)
            for h in recent:
                lines.append(
                    f"{h['round']:<8} {h['my_price']:<14.2f} {h['opp_price']:<14.2f} "
                    f"{h['my_profit']:<14.4f} {h['opp_profit']:.4f}"
                )
        else:
            lines.append("This is the first round. No history yet.")

        lines.append(f"\nValid prices: {[round(p, 3) for p in price_grid]}")
        lines.append("What price do you set this round? Respond with only a number.")
        return "\n".join(lines)

    def choose_price(self, price_grid, history):
        self._price_grid = price_grid
        cost = price_grid[0]

        system = SYSTEM_PROMPT.format(
            cost=cost,
            min_price=round(min(price_grid), 3),
            max_price=round(max(price_grid), 3),
            price_grid=[round(p, 3) for p in price_grid]
        )
        user_msg = self._build_prompt(price_grid,history)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                options={
                    "num_predict": LLM_MAX_TOKENS,
                    "temperature": 0.7,
                },
            )
            raw = response["messages"]["content"].strip()
            chosen = self._parse_price(raw, price_grid)
        except Exception as e:
            print(f"[LLMAgent] Ollama error: {e}, falling back to midpoint price")
            chosen = price_grid[len(price_grid) // 2]

        self.last_price = chosen
        return chosen

    def _parse_price(self, raw, price_grid):
        nums = re.findall(r"\d+\.?\d*", raw)
        if not nums:
            return price_grid[len(price_grid) // 2]

        candidate = float(nums[0])
        closest = min(price_grid, key=lambda p: abs(p - candidate))
        return closest

    def update(self, my_price, opp_price, my_profit, opp_profit) -> None:
        pass