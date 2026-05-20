import logging
import os
import ollama
from agents.base import Agent
from config import LLM_MODEL, LLM_HISTORY_WINDOW, LLM_MAX_TOKENS
from datetime import datetime


def _setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"llm_{timestamp}.log")

    logger = logging.getLogger("llm_agent")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    return logger


logger = _setup_logger()

SYSTEM_PROMPT = """You are Firm A competing in a REPEATED pricing game against Firm B. This game lasts many rounds.

Rules:
- Each round, both firms simultaneously choose a price.
- The firm with the lower price captures the full market (profit = price - {cost}).
- If both choose the same price, each captures half (profit = (price - {cost}) * 0.5).
- You must choose from: {price_grid}
- Price = {cost} yields ZERO profit. If you choose {cost}, you cannot earn money even if you win the entire market. Avoid it unless purely retaliating.

This is a REPEATED game. Think about long-run strategy:
- If your opponent undercuts you repeatedly and you earn zero, you should retaliate by dropping your price.
- If you are both pricing high and earning well, consider holding your price to maintain cooperation.
- Do not anchor on your previous price if it is earning you zero profit.

Think step by step:
1. What has my opponent been doing?
2. Am I earning profit or zero?
3. Should I retaliate, cooperate, or undercut?
4. What price maximizes my long-run profit?

Reply with your reasoning first, then on the final line write: PRICE: <number>"""

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
            is_n_agent = "all_prices" in recent[0] and len(recent[0]["all_prices"]) > 2

            if is_n_agent:
                n = len(recent[0]["all_prices"])
                lines.append(f"Recent rounds ({n}-firm market, most recent last):")
                lines.append(f"{'Round':<8} {'Your Price':<14} {'All Prices':<30} {'Your Profit'}")
                lines.append("-" * 70)
                for h in recent:
                    all_p = [f"{p:.3f}" for p in h["all_prices"]]
                    lines.append(
                        f"{h['round']:<8} {h['my_price']:<14.2f} {str(all_p):<30} {h['my_profit']:.4f}"
                    )

            else:
                lines.append("Recent rounds (most recent last):")
                lines.append(f"{'Round':<8} {'Your Price':<14} {'Opp Price':<14} {'Your Profit':<14} {'Opp Profit'}")
                lines.append("-" * 65)
                for h in recent:
                    lines.append(
                        f"{h['round']:<8} {h['my_price']:<14.3f} {h['opp_price']:<14.3f} "
                        f"{h['my_profit']:<14.4f} {h['opp_profit']:.4f}"
                    )
        else:
            lines.append("This is the first round. No history yet.")

        lines.append(f">>> PICK EXACTLY ONE FROM THESE: {[round(p, 3) for p in price_grid]} <<<")
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
            raw = response["message"]["content"].strip()
            chosen = self._parse_price(raw, price_grid)
            logger.debug(f"\tRound {len(history) + 1}, Model: {self.model}")
            logger.debug(f"PROMPT:\n{user_msg}")
            logger.debug(f"RESPONSE:\n{raw}")
            logger.debug(f"CHOSEN PRICE: {chosen}")
            logger.debug("-" * 65)
        except Exception as e:
            print(f"[LLMAgent] Ollama error: {e}, falling back to midpoint price")
            chosen = price_grid[len(price_grid) // 2]

        self.last_price = chosen
        return chosen

    def _parse_price(self, raw, price_grid):
        import re
        tag_match = re.search(r'PRICE:\s*(\d+\.?\d*)', raw)
        if tag_match:
            candidate = float(tag_match.group(1))
            return min(price_grid, key=lambda p: abs(p - candidate))

        nums = re.findall(r"\d+\.?\d*", raw)
        if not nums:
            return price_grid[len(price_grid) // 2]
        candidate = float(nums[-1])
        return min(price_grid, key=lambda p: abs(p - candidate))

    def update(self, my_profit):
        pass