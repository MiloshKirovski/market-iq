import re
import csv
import os
import sys
import random
import argparse
from pathlib import Path
from collections import Counter

LOG_PATH = Path(os.getenv("LOG_PATH", "logs/llm_20260518_152420.log"))
RAW_CSV = Path("data/llm_traces_raw.csv")
CODED_CSV = Path("data/llm_traces_coded.csv")
REPORT_TXT = Path("data/llm_traces_report.txt")

CATEGORIES = ["cooperation", "retaliation", "undercutting", "neutral"]
EPS = 0.02

RE_HEADER = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - \tRound (\d+), Model: (.+)")
RE_SEPARATOR = re.compile(r"-{30,}")
RE_PRICE_TAG = re.compile(r"CHOSEN PRICE:\s*([\d.]+)")
RE_PROMPT_START = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - PROMPT:")
RE_RESP_START = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - RESPONSE:")
RE_CHOSEN_START = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - CHOSEN PRICE:")
RE_LOG_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - ")
RE_DUO_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", re.MULTILINE)
RE_MULTI_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+\[([^\]]+)\]\s+([\d.]+)", re.MULTILINE)


def _strip_prefix(line):
    return RE_LOG_PREFIX.sub("", line).strip()


def _detect_market_type(prompt_text):
    if any(tok in prompt_text for tok in ("3-firm", "5-firm", "All Prices")):
        return "multi"
    return "duopoly"


def _parse_last_round(prompt_text):
    market_type = _detect_market_type(prompt_text)
    if market_type == "duopoly":
        rows = RE_DUO_ROW.findall(prompt_text)
        if not rows:
            return None, None, None, market_type
        _, your_price, opp_price, your_profit, _ = rows[-1]
        return float(your_price), float(opp_price), float(your_profit), market_type
    else:
        rows = RE_MULTI_ROW.findall(prompt_text)
        if not rows:
            return None, None, None, market_type
        _, your_price, all_prices_str, your_profit = rows[-1]
        prices = [float(p.strip().strip("'\"")) for p in all_prices_str.split(",")]
        your_p = float(your_price)
        other_prices = [p for p in prices if abs(p - your_p) > 0.0001]
        min_opp = min(other_prices) if other_prices else None
        return your_p, min_opp, float(your_profit), market_type


def parse_log(path):
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = text.splitlines()
    traces = []
    i = trace_id = 0

    while i < len(lines):
        m = RE_HEADER.search(lines[i])
        if not m:
            i += 1
            continue

        timestamp = m.group(1)
        round_num = int(m.group(2))
        model = m.group(3).strip()
        i += 1

        prompt_lines, response_lines, chosen_price = [], [], None

        if i < len(lines) and RE_PROMPT_START.search(lines[i]):
            i += 1
            while i < len(lines):
                if RE_RESP_START.search(lines[i]) or RE_CHOSEN_START.search(lines[i]) or RE_HEADER.search(lines[i]):
                    break
                prompt_lines.append(_strip_prefix(lines[i]) if RE_LOG_PREFIX.match(lines[i]) else lines[i])
                i += 1

        if i < len(lines) and RE_RESP_START.search(lines[i]):
            i += 1
            while i < len(lines):
                if RE_CHOSEN_START.search(lines[i]) or RE_HEADER.search(lines[i]) or RE_SEPARATOR.search(
                        _strip_prefix(lines[i])):
                    break
                response_lines.append(_strip_prefix(lines[i]) if RE_LOG_PREFIX.match(lines[i]) else lines[i])
                i += 1

        if i < len(lines) and RE_CHOSEN_START.search(lines[i]):
            m2 = RE_PRICE_TAG.search(lines[i])
            if m2:
                chosen_price = float(m2.group(1))
            i += 1

        prompt_text = "\n".join(prompt_lines).strip()
        response_text = "\n".join(response_lines).strip()

        if not response_text or chosen_price is None:
            continue

        your_prev, opp_prev, your_profit, market_type = _parse_last_round(prompt_text)

        got_zero = (your_profit == 0.0) if your_profit is not None else None
        opp_under = bool(opp_prev is not None and your_prev is not None and got_zero and opp_prev < your_prev)

        if your_prev is not None:
            if chosen_price > your_prev + 0.001:
                direction = "up"
            elif chosen_price < your_prev - 0.001:
                direction = "down"
            else:
                direction = "same"
        else:
            direction = "initial"

        trace_id += 1
        traces.append({
            "trace_id": trace_id,
            "timestamp": timestamp,
            "round": round_num,
            "model": model,
            "market_type": market_type,
            "prompt": prompt_text,
            "response": response_text,
            "chosen_price": chosen_price,
            "prev_your_price": your_prev,
            "prev_opp_price": opp_prev,
            "got_zero_profit": got_zero,
            "opp_undercut": opp_under,
            "price_direction": direction,
        })

    return traces


def run_parse():
    print(f"[parse] Reading {LOG_PATH}")
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    traces = parse_log(LOG_PATH)
    print(f"[parse] Extracted {len(traces)} traces")

    fields = ["trace_id", "timestamp", "round", "model", "market_type", "prompt", "response",
              "chosen_price", "prev_your_price", "prev_opp_price",
              "got_zero_profit", "opp_undercut", "price_direction"]
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(traces)
    print(f"[parse] Saved {RAW_CSV}")
    return traces


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def label_trace(row):
    pm = _to_float(row.get("prev_your_price"))
    po = _to_float(row.get("prev_opp_price"))
    ch = _to_float(row.get("chosen_price"))

    if pm is None or ch is None:
        return "neutral"

    undercut = _to_bool(row.get("opp_undercut"))
    moved_up = ch > pm + EPS
    moved_down = ch < pm - EPS

    if undercut and moved_down:
        return "retaliation"
    if moved_up and po is not None and ch >= po - EPS:
        return "cooperation"
    if po is not None and ch < po - EPS and not undercut:
        return "undercutting"
    return "neutral"


def run_label():
    if not RAW_CSV.exists():
        print("[label] Raw CSV not found - running parse first.")
        run_parse()

    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["label"] = label_trace(row)

    fields = list(rows[0].keys())
    with open(CODED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["label"] for r in rows)
    print(f"[label] Labeled {len(rows)} traces -> {dict(counts)}")
    print(f"[label] Saved {CODED_CSV}")


def _cohen_kappa(labels_a, labels_b):
    cats = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    assert n == len(labels_b), "label lists must be the same length"
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_e = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in cats)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def _dist_block(rows, title):
    lines = [title, "-" * 40]
    total = len(rows)
    cnt = Counter(r["label"] for r in rows)
    for cat in CATEGORIES:
        n = cnt.get(cat, 0)
        pct = 100 * n / total if total else 0
        lines.append(f"  {cat:<14} {n:>6}  ({pct:5.1f}%)")
    return lines


def run_report():
    if not CODED_CSV.exists():
        sys.exit(f"ERROR: {CODED_CSV} not found - run label first.")

    with open(CODED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)

    lines = ["LLM PRICING AGENT - REASONING TRACE REPORT",
             f"Total traces coded: {total}", ""]

    lines += _dist_block(rows, "Behavioral label distribution (all traces):") + [""]

    lines += ["Label distribution by market type:", "-" * 40]
    for mtype in sorted({r["market_type"] for r in rows}):
        sub = [r for r in rows if r["market_type"] == mtype]
        lines.append(f"\n  {mtype}  (n={len(sub)})")
        cnt = Counter(r["label"] for r in sub)
        for cat in CATEGORIES:
            n = cnt.get(cat, 0)
            pct = 100 * n / len(sub) if sub else 0
            lines.append(f"    {cat:<14} {n:>5}  ({pct:5.1f}%)")
    lines.append("")

    lines += ["Response to being undercut (deterministic):", "-" * 40]
    for cond, name in [(True, "WAS undercut last round"),
                       (False, "was NOT undercut")]:
        sub = [r for r in rows if _to_bool(r["opp_undercut"]) == cond
               and r["price_direction"] != "initial"]
        n = len(sub)
        dirs = Counter(r["price_direction"] for r in sub)
        lines.append(f"\n  {name}  (n={n})")
        for d in ("down", "same", "up"):
            k = dirs.get(d, 0)
            lines.append(f"    price {d:<5} {k:>5}  ({100 * k / n if n else 0:5.1f}%)")

    und = [r for r in rows if _to_bool(r["opp_undercut"])]
    below = at = above = 0
    for r in und:
        po, ch = _to_float(r["prev_opp_price"]), _to_float(r["chosen_price"])
        if po is None or ch is None:
            continue
        if ch < po - EPS:
            below += 1
        elif ch > po + EPS:
            above += 1
        else:
            at += 1
    nu = below + at + above
    lines += ["", f"  When undercut, where does the new price land vs opponent's last? (n={nu})"]
    for tag, k in [("below opponent", below), ("at opponent", at), ("above opponent", above)]:
        lines.append(f"    {tag:<16} {k:>5}  ({100 * k / nu if nu else 0:5.1f}%)")
    lines.append("")

    lines += ["Label x price_direction cross-tab:", "-" * 40]
    directions = ["initial", "up", "same", "down"]
    lines.append(f"  {'label':<14}" + "".join(f"{d:>10}" for d in directions))
    for cat in CATEGORIES:
        row_str = f"  {cat:<14}"
        for d in directions:
            k = sum(1 for r in rows if r["label"] == cat and r["price_direction"] == d)
            row_str += f"{k:>10}"
        lines.append(row_str)
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text(report_text, encoding="utf-8")
    print(f"\n[report] Saved {REPORT_TXT}")


def main():
    parser = argparse.ArgumentParser(description="LLM Pricing Agent Trace Pipeline")
    parser.add_argument("--step", choices=["parse", "label", "report", "all"],
                        default="all", help="Which step to run (default: all)")
    args = parser.parse_args()

    # python pipeline.py --step all
    # python pipeline.py --step parse
    # python pipeline.py --step label
    # python pipeline.py --step report
    if args.step in ("parse", "all"): run_parse()
    if args.step in ("label", "all"): run_label()
    if args.step in ("report", "all"): run_report()


if __name__ == "__main__":
    main()