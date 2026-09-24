#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WinGo Predictor Engine v5.0 — Proxy-Enabled Edition
- 14-algorithm ensemble
- Free proxy auto-rotation
- Chrome impersonation via curl_cffi
"""

import csv
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from statistics import mean, stdev

try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_OK = True
except ImportError:
    CURL_CFFI_OK = False
    curl_requests = None

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════
API_ENDPOINTS = [
    "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
    "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json",
]

PROXY_LIST_URLS = [
    "https://cdn.jsdelivr.net/gh/proxyscrape/free-proxy-list@main/proxies/protocols/http/data.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
]

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://draw.ar-lottery01.com",
    "referer": "https://draw.ar-lottery01.com/",
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
}

DATA_DIR = os.environ.get("WINGO_DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATA_DIR, "wingo_data.csv")
PRED_LOG = os.path.join(DATA_DIR, "pred_log.csv")
LOG_FILE = os.path.join(DATA_DIR, "wingo_predictor.log")

MIN_DATA = 50
MAX_RETRIES = 2
RETRY_DELAY = 2
MIN_CONSENSUS = 3
CALIB_WINDOW = 30
FETCH_TIMEOUT = 12
PROXY_TEST_TIMEOUT = 8
MAX_PROXY_TRIES = 25

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wingo")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
log.addHandler(console)


# ════════════════════════════════════════════════════════════
#  MATH
# ════════════════════════════════════════════════════════════
def sigmoid_conf(raw, k=0.08, midpoint=60):
    try:
        return round(min(50 + 47 / (1 + math.exp(-k * (raw - midpoint))), 97), 1)
    except OverflowError:
        return 97.0


def entropy(seq):
    if not seq:
        return 0.0
    total = len(seq)
    counts = Counter(seq)
    return -sum((v / total) * math.log2(v / total)
                for v in counts.values() if v)


def ema(values, period):
    if not values:
        return 0.0
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


# ════════════════════════════════════════════════════════════
#  CSV
# ════════════════════════════════════════════════════════════
def issue_key(issue):
    try:
        return (0, int(str(issue)))
    except (ValueError, TypeError):
        return (1, str(issue))


def save_to_csv(issue, num, size):
    existing = set()
    if os.path.isfile(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE, dtype={"Issue": str})
            existing = set(df["Issue"].astype(str).tolist())
        except Exception:
            pass
    if str(issue) in existing:
        return False
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["Issue", "Number", "Size", "Time"])
            w.writerow([issue, num, size,
                        time.strftime("%Y-%m-%d %H:%M:%S")])
        return True
    except Exception as e:
        log.warning("CSV write failed: %s", e)
        return False


def load_data():
    cols = ["Issue", "Number", "Size", "Time"]
    if not os.path.isfile(CSV_FILE):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(CSV_FILE, dtype={"Issue": str})
        if df.empty:
            return pd.DataFrame(columns=cols)
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
        df.dropna(subset=["Number"], inplace=True)
        df["Number"] = df["Number"].astype(int)
        df["Size"] = df["Number"].apply(lambda n: "Big" if n >= 5 else "Small")
        df = df.drop_duplicates(subset="Issue")
        df = df.iloc[df["Issue"].map(issue_key).argsort(kind="stable")]
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.warning("CSV load error: %s", e)
        return pd.DataFrame(columns=cols)


def log_prediction(issue, pred, conf, actual, hit):
    try:
        file_exists = os.path.isfile(PRED_LOG)
        with open(PRED_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(["Issue", "Pred", "Conf", "Actual", "Hit", "Time"])
            w.writerow([issue, pred, conf, actual, int(bool(hit)),
                        time.strftime("%Y-%m-%d %H:%M:%S")])
    except Exception as e:
        log.warning("pred_log write failed: %s", e)


# ════════════════════════════════════════════════════════════
#  SESSION
# ════════════════════════════════════════════════════════════
class AlgoTracker:
    def __init__(self, window=CALIB_WINDOW):
        self.window = window
        self.records = deque(maxlen=window)

    def update(self, algo_name, hit):
        self.records.append((algo_name, bool(hit)))

    def accuracy(self, algo_name):
        recs = [h for a, h in self.records if a == algo_name]
        if len(recs) < 5:
            return None
        return sum(recs) / len(recs)

    def dynamic_weight(self, base, algo_name):
        acc = self.accuracy(algo_name)
        if acc is None:
            return base
        return base * (0.5 + acc)


class SessionState:
    def __init__(self):
        self.prior_big = 0.50
        self.total_pred = 0
        self.correct = 0
        self.history = deque(maxlen=200)
        self.alpha = 1
        self.beta_param = 1
        self.tracker = AlgoTracker()
        self.streak = 0
        self.best_streak = 0

    def update(self, pred, actual, votes=None):
        self.total_pred += 1
        hit = (pred == actual)
        if hit:
            self.correct += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            self.streak = 0
        self.history.append((pred, actual))
        if actual == "Big":
            self.alpha += 1
        else:
            self.beta_param += 1
        self.prior_big = self.alpha / (self.alpha + self.beta_param)
        if votes:
            for name, v in votes.items():
                self.tracker.update(name, v["pred"] == actual)

    @property
    def accuracy(self):
        return round(self.correct / self.total_pred * 100, 1) \
            if self.total_pred else 0.0

    @property
    def recent_accuracy(self):
        recent = list(self.history)[-20:]
        if not recent:
            return 0.0
        return round(sum(1 for p, a in recent if p == a)
                     / len(recent) * 100, 1)


# ════════════════════════════════════════════════════════════
#  ALGORITHMS
# ════════════════════════════════════════════════════════════
def _avg_streak_len(sizes):
    if len(sizes) < 4:
        return 3.0
    streaks, run = [], 1
    for i in range(1, len(sizes)):
        if sizes[i] == sizes[i - 1]:
            run += 1
        else:
            streaks.append(run)
            run = 1
    streaks.append(run)
    return mean(streaks) if streaks else 3.0


def algo_dragon(sizes):
    if len(sizes) < 5:
        return None, 0, ""
    streak, current = 1, sizes[-1]
    for i in range(len(sizes) - 1, 0, -1):
        if sizes[i] == sizes[i - 1]:
            streak += 1
        else:
            break
    avg_streak = _avg_streak_len(sizes[:-streak] if streak < len(sizes) else [])
    if streak >= max(6, avg_streak + 2):
        pred = "Big" if current == "Small" else "Small"
        return pred, sigmoid_conf(60 + streak * 2), \
            f"Long streak break ({streak}x {current})"
    if streak >= 3:
        return current, sigmoid_conf(55 + streak * 3), \
            f"Dragon continue ({streak}x {current})"
    pred = "Big" if current == "Small" else "Small"
    return pred, sigmoid_conf(52), "Short streak reversal"


def algo_balance(df):
    if len(df) < 30:
        return None, 0, ""
    windows = [(df.tail(100), 0.40), (df.tail(50), 0.35),
               (df.tail(20), 0.25)]
    big_score = small_score = 0.0
    for win, weight in windows:
        b = win["Size"].value_counts().get("Big", 0)
        s = win["Size"].value_counts().get("Small", 0)
        t = b + s
        if not t:
            continue
        bp, sp = b / t * 100, s / t * 100
        if bp > 55:
            small_score += (bp - 50) * weight
        elif sp > 55:
            big_score += (sp - 50) * weight
    if big_score > small_score and big_score > 3:
        return "Big", sigmoid_conf(50 + big_score * 2), \
            f"Small overload ({big_score:.1f})"
    if small_score > big_score and small_score > 3:
        return "Small", sigmoid_conf(50 + small_score * 2), \
            f"Big overload ({small_score:.1f})"
    return None, 0, ""


def algo_momentum(df):
    if len(df) < 20:
        return None, 0, ""
    def sw(sizes, w):
        b = sizes.count("Big")
        t = len(sizes)
        return ((b - (t - b)) / t) * w if t else 0
    l5 = df.tail(5)["Size"].tolist()
    l10 = df.tail(10)["Size"].tolist()
    l20 = df.tail(20)["Size"].tolist()
    m = sw(l5, .5) + sw(l10, .3) + sw(l20, .2)
    if m > 0.15:
        return "Big", sigmoid_conf(50 + abs(m) * 80), f"Momentum Big ({m:.2f})"
    if m < -0.15:
        return "Small", sigmoid_conf(50 + abs(m) * 80), f"Momentum Small ({m:.2f})"
    return None, 0, ""


def algo_pattern(df):
    if len(df) < 20:
        return None, 0, ""
    sizes = df["Size"].tolist()
    best_pred, best_conf, best_reason = None, 0, ""
    for order in (4, 3):
        if len(sizes) < order + 1:
            continue
        last_n = sizes[-order:]
        nxt = [sizes[i + order]
               for i in range(len(sizes) - order - 1)
               if sizes[i:i + order] == last_n]
        if len(nxt) < 3:
            continue
        b, t = nxt.count("Big"), len(nxt)
        if b / t >= 0.60:
            conf = sigmoid_conf(50 + (b / t - .5) * 120)
            if conf > best_conf:
                best_pred, best_conf, best_reason = "Big", conf, \
                    f"Seq-{order}: Big {b}/{t}"
        elif (t - b) / t >= 0.60:
            conf = sigmoid_conf(50 + ((t - b) / t - .5) * 120)
            if conf > best_conf:
                best_pred, best_conf, best_reason = "Small", conf, \
                    f"Seq-{order}: Small {t-b}/{t}"
    return best_pred, best_conf, best_reason


def algo_cluster(df):
    if len(df) < 25:
        return None, 0, ""
    counter = Counter(df.tail(20)["Number"].tolist())
    num, freq = counter.most_common(1)[0]
    if freq < 3:
        return None, 0, ""
    hot_side = "Big" if num >= 5 else "Small"
    if freq >= 5:
        return ("Big" if hot_side == "Small" else "Small",
                sigmoid_conf(52 + freq * 3),
                f"Hot#{int(num)} ({freq}x) cooldown")
    return hot_side, sigmoid_conf(53 + freq * 2), \
        f"Hot#{int(num)} ({freq}x) continuing"


def algo_volatility(df):
    if len(df) < 15:
        return None, 0, ""
    nums = df.tail(10)["Number"].tolist()
    avg = mean(nums)
    try:
        sd = stdev(nums)
    except Exception:
        return None, 0, ""
    if sd < 1.5:
        return ("Big" if avg > 4.5 else "Small"), sigmoid_conf(60), \
            f"Low vol (SD {sd:.1f})"
    if sd > 3.0:
        return ("Big" if avg < 4.5 else "Small"), sigmoid_conf(58), \
            f"High vol reversal (SD {sd:.1f})"
    return None, 0, ""


def algo_parity(df):
    if len(df) < 20:
        return None, 0, ""
    last10 = df.tail(10)["Number"].tolist()
    odds = sum(1 for n in last10 if int(n) % 2 == 1)
    evens = 10 - odds
    avg = mean(last10)
    if odds >= 8:
        pred = "Big" if avg < 4.5 else "Small"
        return pred, sigmoid_conf(58), f"Odd surge ({odds}/10)"
    if evens >= 8:
        pred = "Small" if avg >= 4.5 else "Big"
        return pred, sigmoid_conf(58), f"Even surge ({evens}/10)"
    return None, 0, ""


def algo_zone(df):
    if len(df) < 20:
        return None, 0, ""
    nums = df.tail(15)["Number"].tolist()
    lo = sum(1 for n in nums if n <= 2)
    hi = sum(1 for n in nums if n >= 7)
    total = len(nums)
    if lo / total >= 0.45:
        return "Big", sigmoid_conf(57), f"Low-zone ({lo}/15)"
    if hi / total >= 0.45:
        return "Small", sigmoid_conf(57), f"High-zone ({hi}/15)"
    return None, 0, ""


def algo_markov(df):
    if len(df) < 40:
        return None, 0, ""
    sizes = df["Size"].tolist()
    trans = defaultdict(Counter)
    for i in range(2, len(sizes)):
        state = (sizes[i - 2], sizes[i - 1])
        trans[state][sizes[i]] += 1
    state_now = (sizes[-2], sizes[-1])
    counts = trans.get(state_now)
    if not counts:
        return None, 0, ""
    total = sum(counts.values())
    if total < 5:
        return None, 0, ""
    best_pred = max(counts, key=counts.get)
    prob = counts[best_pred] / total
    if prob < 0.58:
        return None, 0, ""
    conf = sigmoid_conf(50 + (prob - 0.5) * 120)
    return best_pred, conf, f"Markov {state_now}->{best_pred} ({prob:.0%})"


def algo_entropy(df):
    if len(df) < 30:
        return None, 0, ""
    sizes = df.tail(20)["Size"].tolist()
    H = entropy(sizes)
    recent = df.tail(5)["Size"].tolist()
    pred_side, _ = Counter(recent).most_common(1)[0]
    if H < 0.75:
        conf = sigmoid_conf(55 + (0.75 - H) * 60)
        return pred_side, conf, f"Low entropy ({H:.2f})"
    return None, 0, ""


def algo_gap(df):
    if len(df) < 30:
        return None, 0, ""
    nums = df["Number"].tolist()
    last, total = {}, len(nums)
    for i, n in enumerate(nums):
        last[int(n)] = i
    overdue_big = sum(total - last.get(n, 0) for n in range(5, 10))
    overdue_small = sum(total - last.get(n, 0) for n in range(0, 5))
    diff = overdue_big - overdue_small
    if diff > total * 0.5:
        return "Big", sigmoid_conf(54 + diff / total * 10), \
            f"Big overdue (diff {diff})"
    if diff < -total * 0.5:
        return "Small", sigmoid_conf(54 + abs(diff) / total * 10), \
            f"Small overdue (diff {-diff})"
    return None, 0, ""


def algo_session_bias(df, prior_big):
    if len(df) < 30:
        return None, 0, ""
    tail30 = df.tail(30)
    b30 = tail30["Size"].value_counts().get("Big", 0)
    s30 = tail30["Size"].value_counts().get("Small", 0)
    total = b30 + s30
    if not total:
        return None, 0, ""
    local = b30 / total
    if local >= 0.60 and prior_big >= 0.55:
        return "Big", sigmoid_conf(52 + (local - 0.5) * 80), \
            f"Session Big ({local:.0%})"
    if local <= 0.40 and prior_big <= 0.45:
        return "Small", sigmoid_conf(52 + (0.5 - local) * 80), \
            f"Session Small ({local:.0%})"
    return None, 0, ""


def algo_conditional(df):
    if len(df) < 40:
        return None, 0, ""
    sizes = df["Size"].tolist()
    counts = {"Big": Counter(), "Small": Counter()}
    for i in range(1, len(sizes)):
        counts[sizes[i - 1]][sizes[i]] += 1
    last = sizes[-1]
    row = counts[last]
    total = sum(row.values())
    if total < 10:
        return None, 0, ""
    best = max(row, key=row.get)
    prob = row[best] / total
    if prob < 0.56:
        return None, 0, ""
    conf = sigmoid_conf(50 + (prob - 0.5) * 100)
    return best, conf, f"P({best}|{last})={prob:.0%}"


def algo_ema(df):
    if len(df) < 25:
        return None, 0, ""
    nums = df["Number"].tolist()
    fast = ema(nums[-15:], 5)
    slow = ema(nums[-25:], 12)
    if abs(fast - slow) < 0.35:
        return None, 0, ""
    pred = "Big" if fast > slow else "Small"
    conf = sigmoid_conf(53 + abs(fast - slow) * 6)
    return pred, conf, f"EMA fast={fast:.2f} slow={slow:.2f}"


# ════════════════════════════════════════════════════════════
#  VOTING ENGINE
# ════════════════════════════════════════════════════════════
ALGO_WEIGHTS = {
    "Dragon": 1.4, "Balance": 1.3, "Momentum": 1.2, "Pattern": 1.6,
    "Cluster": 1.0, "Volatility": 0.9, "Parity": 1.0, "Zone": 0.9,
    "Markov": 1.7, "Entropy": 1.1, "Gap": 0.8, "SessionBias": 1.2,
    "Conditional": 1.5, "EMA": 1.3,
}
TOTAL_ALGOS = len(ALGO_WEIGHTS)
ALGO_ORDER = list(ALGO_WEIGHTS.keys())


def run_algorithms(df, prior_big=0.5):
    return {
        "Dragon": algo_dragon(df["Size"].tolist()),
        "Balance": algo_balance(df),
        "Momentum": algo_momentum(df),
        "Pattern": algo_pattern(df),
        "Cluster": algo_cluster(df),
        "Volatility": algo_volatility(df),
        "Parity": algo_parity(df),
        "Zone": algo_zone(df),
        "Markov": algo_markov(df),
        "Entropy": algo_entropy(df),
        "Gap": algo_gap(df),
        "SessionBias": algo_session_bias(df, prior_big),
        "Conditional": algo_conditional(df),
        "EMA": algo_ema(df),
    }


def multi_layer_prediction(df, session=None):
    total = len(df)
    if total < MIN_DATA:
        return {"pred": "WAIT", "conf": 0, "final_confidence": 0,
                "logic": f"Collecting... {total}/{MIN_DATA}",
                "stats": "-", "votes": {}, "total_data": total,
                "win_rate": 0.0, "needed": MIN_DATA - total,
                "agreement": "0/0", "entropy": 0.0,
                "big_score": 0, "small_score": 0,
                "bayesian_prior": 50.0}

    prior_big = session.prior_big if session else 0.5
    algos = run_algorithms(df, prior_big)

    big_score = small_score = 0.0
    vote_log, active = {}, 0

    for name in ALGO_ORDER:
        res = algos.get(name)
        if not res or len(res) < 2:
            continue
        pred, conf, *reason = res
        if not pred or conf == 0:
            continue
        reason_str = reason[0] if reason else ""
        active += 1
        base_w = ALGO_WEIGHTS[name]
        w = session.tracker.dynamic_weight(base_w, name) if session else base_w
        wc = conf * w
        vote_log[name] = {"pred": pred, "conf": conf, "w": round(w, 2),
                          "reason": reason_str}
        if pred == "Big":
            big_score += wc
        else:
            small_score += wc

    if big_score == 0 and small_score == 0:
        return {"pred": "WAIT", "conf": 50, "final_confidence": 50,
                "logic": "No strong signal", "stats": "-",
                "votes": vote_log, "total_data": total, "win_rate": 0.0,
                "agreement": "0/0", "entropy": 0.0,
                "big_score": 0, "small_score": 0, "bayesian_prior": 50.0}

    tot_score = big_score + small_score
    final_pred = "Big" if big_score > small_score else "Small"
    raw_conf = (max(big_score, small_score) / tot_score) * 100

    bv = sum(1 for v in vote_log.values() if v["pred"] == "Big")
    sv = len(vote_log) - bv
    tv = len(vote_log)
    votes_for_winner = bv if final_pred == "Big" else sv

    if votes_for_winner < MIN_CONSENSUS:
        return {"pred": "WAIT", "conf": raw_conf,
                "final_confidence": raw_conf,
                "logic": f"Low consensus ({votes_for_winner}/{tv})",
                "stats": "-", "votes": vote_log, "total_data": total,
                "win_rate": 0.0, "agreement": f"{votes_for_winner}/{tv}",
                "needed": None, "entropy": 0.0,
                "big_score": round(big_score, 1),
                "small_score": round(small_score, 1),
                "bayesian_prior": round(prior_big * 100, 1)}

    ab = ((max(bv, sv) / tv) - 0.5) * 20 if tv else 0
    prior_factor = (prior_big - 0.5) * 6
    bayesian_adj = prior_factor if final_pred == "Big" else -prior_factor
    final_conf = sigmoid_conf(raw_conf + ab + bayesian_adj)

    dominant = max(
        (k for k, v in vote_log.items() if v["pred"] == final_pred),
        key=lambda k: vote_log[k]["conf"], default="Combined")
    main_logic = vote_log.get(dominant, {}).get("reason",
                                                 "Multi-algo consensus")

    b100 = df.tail(100)["Size"].value_counts().get("Big", 0)
    s100 = df.tail(100)["Size"].value_counts().get("Small", 0)
    b20 = df.tail(20)["Size"].value_counts().get("Big", 0)
    s20 = df.tail(20)["Size"].value_counts().get("Small", 0)
    H = round(entropy(df.tail(20)["Size"].tolist()), 3)
    stats_str = (f"100G B{b100}/S{s100} | 20G B{b20}/S{s20} "
                 f"| Algos {active}/{TOTAL_ALGOS} | H={H}")

    return {
        "pred": final_pred,
        "conf": round(raw_conf, 1),
        "final_confidence": final_conf,
        "logic": f"[{dominant}] {main_logic}",
        "stats": stats_str,
        "votes": vote_log,
        "total_data": total,
        "win_rate": 0.0,
        "big_score": round(big_score, 1),
        "small_score": round(small_score, 1),
        "agreement": f"{votes_for_winner}/{tv}",
        "entropy": H,
        "bayesian_prior": round(prior_big * 100, 1),
    }


# ════════════════════════════════════════════════════════════
#  NETWORK — FREE PROXY AUTO-ROTATION
# ════════════════════════════════════════════════════════════
_CACHED_PROXY = None
_PROXY_FETCH_TIME = 0


def fetch_proxy_list():
    """Fetch free proxy list from multiple sources."""
    all_proxies = set()
    for url in PROXY_LIST_URLS:
        try:
            log.info("Fetching proxies from: %s", url[:70])
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                lines = [l.strip() for l in r.text.splitlines()
                         if l.strip() and not l.startswith("#")]
                for line in lines:
                    if ":" in line and len(line) < 30:
                        all_proxies.add(line.strip())
                log.info("  Got %d proxies (total: %d)",
                         len(lines), len(all_proxies))
        except Exception as e:
            log.warning("  Failed: %s", e)

    proxies = list(all_proxies)
    log.info("Total unique proxies: %d", len(proxies))
    return proxies


def test_proxy(proxy):
    """Test if a proxy can reach the API."""
    try:
        session = curl_requests.Session(impersonate="chrome124")
        session.proxies = {
            "http": f"http://{proxy}",
            "https": f"http://{proxy}"
        }
        url = API_ENDPOINTS[0] + f"?ts={int(time.time()*1000)}"
        resp = session.get(url, timeout=PROXY_TEST_TIMEOUT)

        if resp.status_code == 200:
            try:
                data = resp.json()
                lst = _extract_list(data)
                if lst:
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def get_working_proxy():
    """Get a working proxy from the pool (cached for 5 minutes)."""
    global _CACHED_PROXY, _PROXY_FETCH_TIME

    if _CACHED_PROXY and (time.time() - _PROXY_FETCH_TIME) < 300:
        return _CACHED_PROXY

    log.info("=" * 60)
    log.info("Searching for working proxy...")
    log.info("=" * 60)

    proxies = fetch_proxy_list()
    if not proxies:
        log.warning("No proxies available")
        return None

    random.shuffle(proxies)
    tried = 0

    for proxy in proxies:
        if tried >= MAX_PROXY_TRIES:
            break
        tried += 1
        log.info("Testing proxy %d/%d: %s",
                 tried, min(MAX_PROXY_TRIES, len(proxies)), proxy)
        if test_proxy(proxy):
            log.info("✓ FOUND WORKING PROXY: %s", proxy)
            _CACHED_PROXY = proxy
            _PROXY_FETCH_TIME = time.time()
            return proxy

    log.warning("No working proxy found after %d tries", tried)
    return None


def _extract_list(data):
    """Extract draw list from API response."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("list", "data", "items", "result", "results", "rows"):
        v = data.get(key)
        if isinstance(v, list):
            return v
    d = data.get("data")
    if isinstance(d, dict):
        for key in ("list", "data", "items", "result", "results", "rows"):
            v = d.get(key)
            if isinstance(v, list):
                return v
    return []


def _try_fetch(session, url_base):
    """Try to fetch from one endpoint with given session."""
    try:
        url = f"{url_base}?ts={int(time.time() * 1000)}"
        log.info("Fetching: %s", url[:80])
        resp = session.get(url, timeout=FETCH_TIMEOUT)
        log.info("  status=%s len=%s", resp.status_code, len(resp.text))

        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception as je:
            log.warning("  JSON decode failed: %s", je)
            return None

        lst = _extract_list(data)
        if lst:
            log.info("  ✓ GOT %d items", len(lst))
            return lst
        log.warning("  ✗ empty list, keys=%s",
                    list(data.keys()) if isinstance(data, dict) else "?")
        return None
    except Exception as e:
        log.warning("  ✗ exception: %s", e)
        return None


def fetch_latest():
    """Fetch latest draw with proxy rotation."""
    log.info("=" * 60)
    log.info("fetch_latest() START")
    log.info("=" * 60)

    working_proxy = get_working_proxy()

    # Try with proxy first
    if working_proxy:
        try:
            session = curl_requests.Session(impersonate="chrome124")
            session.headers.update(HEADERS)
            session.proxies = {
                "http": f"http://{working_proxy}",
                "https": f"http://{working_proxy}"
            }
            for url_base in API_ENDPOINTS:
                lst = _try_fetch(session, url_base)
                if lst:
                    return lst[0]
        except Exception as e:
            log.warning("Proxy session failed: %s", e)

    # Fallback: direct (no proxy)
    log.info("Trying direct (no proxy)...")
    for url_base in API_ENDPOINTS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                session = curl_requests.Session(impersonate="chrome124")
                session.headers.update(HEADERS)
                lst = _try_fetch(session, url_base)
                if lst:
                    return lst[0]
            except Exception as e:
                log.warning("Direct attempt %d failed: %s", attempt, e)
            time.sleep(RETRY_DELAY)

    log.error("All attempts failed")
    return None


def fetch_history_pages(pages=4):
    """Fetch history pages with proxy."""
    log.info("=" * 60)
    log.info("fetch_history_pages() START")
    log.info("=" * 60)

    working_proxy = get_working_proxy()
    results = []

    if working_proxy:
        try:
            session = curl_requests.Session(impersonate="chrome124")
            session.headers.update(HEADERS)
            session.proxies = {
                "http": f"http://{working_proxy}",
                "https": f"http://{working_proxy}"
            }
            for url_base in API_ENDPOINTS:
                for page in range(1, pages + 1):
                    try:
                        url = (f"{url_base}?ts={int(time.time()*1000)}"
                               f"&pageNo={page}&pageSize=100")
                        resp = session.get(url, timeout=FETCH_TIMEOUT)
                        if resp.status_code == 200:
                            lst = _extract_list(resp.json())
                            if lst:
                                results.extend(lst)
                                log.info("History p%d: %d items",
                                         page, len(lst))
                        time.sleep(0.5)
                    except Exception as e:
                        log.warning("History page %d failed: %s", page, e)
                if results:
                    log.info("Total history: %d items", len(results))
                    return results
        except Exception as e:
            log.warning("Proxy history fetch failed: %s", e)

    log.warning("History fetch returned %d items", len(results))
    return results
