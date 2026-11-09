"""Realistic synthetic UPI ledger with structured fraud (v2).

Replaces the random fraud injection. What is different:
  * Accounts mostly have their own device; a few family pairs legitimately share one.
  * Each account pays a small set of known contacts. Popular merchants get many payers (legit hubs).
  * RAPID TRANSFER  : bursts of 5 payments in minutes, from accounts in 4 rings that share one device and
                      pay the same fresh "mule" beneficiary. (Velocity is kept, not overwritten.)
  * NEW BENEFICIARY : large payments to a very new beneficiary, often from an unfamiliar device/location.
  * SUSPICIOUS COLLECT: collect requests with ordinary amounts (behaviourally normal; the message and the
                      network are what expose them), mostly paid to mule beneficiaries.
  * Hard negatives  : legitimate large rent-style payments to new people, legitimate shopping bursts,
                      legitimate night payments, legitimate collect requests, legitimate shared devices.

Writes accounts.csv, beneficiaries.csv, transactions.csv, fraud_cases.csv into data/synthetic/.
Run messages generator AFTER this one.     python data_generation/generate_transactions_v2.py
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
rng = random.Random(SEED)
nrng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT = BASE_DIR / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

N_ACCOUNTS, N_BEN, N_TOTAL = 1000, 2000, 5000
START, END = datetime(2026, 8, 10), datetime(2026, 11, 7)
N_RINGS, RING_SIZE, N_BURSTS, BURST_LEN = 4, 5, 18, 5      # 90 rapid-transfer payments
N_NEWBEN, N_COLLECT = 80, 80
N_LEGIT_LARGE, N_LEGIT_BURSTS, LEGIT_BURST_LEN = 60, 20, 3
N_MULES = 8
HOUR_W = [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9, 9, 8, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2]  # day-heavy


def rand_time(night=False):
    day = START + timedelta(days=rng.random() * (END - START).days)
    hour = rng.choice([23, 0, 1, 2, 3, 4, 5]) if night else rng.choices(range(24), weights=HOUR_W)[0]
    return day.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59), microsecond=0)


# ---------------------------------------------------------------- accounts
acc_ids = [f"ACC_{i:04d}" for i in range(1, N_ACCOUNTS + 1)]
accounts = pd.DataFrame({
    "account_id": acc_ids,
    "account_age_days": np.where(nrng.random(N_ACCOUNTS) < 0.08, nrng.integers(10, 60, N_ACCOUNTS),
                                 nrng.integers(60, 1500, N_ACCOUNTS)),
    "device_id": [f"DEV_{i:04d}" for i in range(1, N_ACCOUNTS + 1)],
    "normal_daily_txn_count": nrng.integers(1, 9, N_ACCOUNTS),
    "average_transaction_amount": np.clip(np.exp(nrng.normal(np.log(900), 0.8, N_ACCOUNTS)), 150, 8000).round(2),
    "risk_profile": "normal",
    "home_location_id": [f"LOC_{rng.randint(1, 50):03d}" for _ in range(N_ACCOUNTS)],
    "account_balance": nrng.uniform(8000, 120000, N_ACCOUNTS).round(2),
})
pool = acc_ids[:]
rng.shuffle(pool)
ring_members = [pool[r * RING_SIZE:(r + 1) * RING_SIZE] for r in range(N_RINGS)]
ring_device = {r: f"DEV_{9001 + r}" for r in range(N_RINGS)}
ring_accounts = {a for m in ring_members for a in m}
for r, members in enumerate(ring_members):                       # a ring shares ONE device
    accounts.loc[accounts.account_id.isin(members), "device_id"] = ring_device[r]
family = [a for a in pool[N_RINGS * RING_SIZE:] ][:30]
for i in range(0, 30, 2):                                        # 15 legitimate family pairs share a device
    accounts.loc[accounts.account_id == family[i + 1], "device_id"] = \
        accounts.loc[accounts.account_id == family[i], "device_id"].iloc[0]
acc = accounts.set_index("account_id")

# ---------------------------------------------------------- beneficiaries
cats = nrng.choice(["merchant", "p2p_individual", "utility", "unknown"], N_BEN, p=[.40, .35, .15, .10])
ages = np.array([rng.randint(200, 1500) if c in ("merchant", "utility") else
                 (rng.randint(10, 700) if c == "p2p_individual" else rng.randint(1, 300)) for c in cats])
ben = pd.DataFrame({"beneficiary_id": [f"BEN_{i:04d}" for i in range(1, N_BEN + 1)],
                    "beneficiary_age_days": ages, "category": cats})
cand = ben.index[ben.category.isin(["p2p_individual", "unknown"])].tolist()
mule_idx = rng.sample(cand, N_MULES)
for i in mule_idx:
    ben.loc[i, "beneficiary_age_days"] = rng.randint(1, 9)
mules = ben.loc[mule_idx, "beneficiary_id"].tolist()
ben_i = ben.set_index("beneficiary_id")
non_mule = ben[~ben.beneficiary_id.isin(mules)]
# popularity: merchants/utilities attract many payers (legitimate high in-degree)
w = np.where(non_mule.category.isin(["merchant", "utility"]), nrng.lognormal(0, 1.0, len(non_mule)), 0.25)
w = w / w.sum()
contacts = {a: list(nrng.choice(non_mule.beneficiary_id, size=rng.randint(5, 12), replace=False, p=w))
            for a in acc_ids}

rows = []


def add(sender, receiver, ts, amount, *, ttype="P2P", channel="UPI", device=None, loc=None, new=0,
        fraud=0, ftype="normal", sid=""):
    a = acc.loc[sender]
    rows.append(dict(
        timestamp=ts, sender_id=sender, receiver_id=receiver, amount=round(float(amount), 2),
        transaction_type=ttype, payment_channel=channel, device_id=device or a.device_id,
        location_id=loc or a.home_location_id, sender_balance=round(float(a.account_balance * rng.uniform(.85, 1.15)), 2),
        account_age_days=int(a.account_age_days), beneficiary_age_days=int(ben_i.loc[receiver, "beneficiary_age_days"]),
        is_new_beneficiary=new, is_fraud=fraud, fraud_type=ftype, scenario_id=sid))


def other_loc(sender):
    return rng.choice([f"LOC_{i:03d}" for i in range(1, 51) if f"LOC_{i:03d}" != acc.loc[sender].home_location_id])


def kind_for(receiver):
    return ("P2M", "QR" if rng.random() < .4 else "UPI") if ben_i.loc[receiver, "category"] in ("merchant", "utility") \
        else ("P2P", "UPI")


# ------------------------------------------------------------------- fraud
for b in range(N_BURSTS):                                           # RAPID TRANSFER from rings
    r = b % N_RINGS
    sender = rng.choice(ring_members[r])
    mule = mules[r]
    t0 = rand_time(night=rng.random() < 0.65)
    for k in range(BURST_LEN):
        add(sender, mule, t0 + timedelta(seconds=k * rng.randint(30, 110)),
            acc.loc[sender].average_transaction_amount * rng.uniform(.8, 2.0), new=1, fraud=1, ftype="rapid_transfer",
            loc=other_loc(sender) if rng.random() < .7 else None, sid=f"RAPID_{b:02d}")
victims = [a for a in acc_ids if a not in ring_accounts]
for i in range(N_NEWBEN):                                           # NEW-BENEFICIARY ANOMALY
    s = rng.choice(victims)
    young = ben[(ben.beneficiary_age_days < 20) & ~ben.beneficiary_id.isin(mules)].beneficiary_id.tolist()
    recv = rng.choice(mules) if rng.random() < .7 else rng.choice(young)
    amt = min(acc.loc[s].average_transaction_amount * rng.uniform(5, 12), acc.loc[s].account_balance * .9)
    add(s, recv, rand_time(night=rng.random() < .4), amt, new=1, fraud=1, ftype="new_beneficiary_anomaly",
        device=f"DEV_N{rng.randint(1000, 9999)}" if rng.random() < .5 else None,
        loc=other_loc(s) if rng.random() < .6 else None, sid=f"NEWBEN_{i:03d}")
for i in range(N_COLLECT):                                          # SUSPICIOUS COLLECT (ordinary amounts)
    s = rng.choice(victims)
    recv = rng.choice(mules) if rng.random() < .7 else rng.choice(
        ben[(ben.beneficiary_age_days < 30) & ~ben.beneficiary_id.isin(mules)].beneficiary_id.tolist())
    add(s, recv, rand_time(), acc.loc[s].average_transaction_amount * rng.uniform(.8, 3), ttype="collect_request",
        channel="collect", new=1, fraud=1, ftype="suspicious_collect", sid=f"COLLECT_{i:03d}")

# --------------------------------------------------- legitimate hard negatives
for i in range(N_LEGIT_LARGE):                                      # rent / deposit style large payments
    s = rng.choice(acc_ids)
    recv = rng.choice(non_mule[non_mule.category == "p2p_individual"].beneficiary_id.tolist())
    amt = min(acc.loc[s].average_transaction_amount * rng.uniform(4, 9), acc.loc[s].account_balance * .8)
    add(s, recv, rand_time(), amt, new=1, sid="LEGIT_LARGE")
for i in range(N_LEGIT_BURSTS):                                     # festive shopping burst, known merchants
    s = rng.choice(acc_ids)
    t0 = rand_time()
    for k in range(LEGIT_BURST_LEN):
        recv = rng.choice(contacts[s])
        tt, ch = kind_for(recv)
        add(s, recv, t0 + timedelta(minutes=k * rng.randint(8, 20)),
            acc.loc[s].average_transaction_amount * rng.uniform(.5, 1.6), ttype=tt, channel=ch, sid="LEGIT_BURST")

# ------------------------------------------------------------ ordinary legit
weights = acc.normal_daily_txn_count.values / acc.normal_daily_txn_count.sum()
while len(rows) < N_TOTAL:
    s = str(nrng.choice(acc_ids, p=weights))
    a = acc.loc[s]
    is_new = rng.random() < .12
    recv = rng.choice(non_mule.beneficiary_id.tolist()) if is_new else rng.choice(contacts[s])
    tt, ch = kind_for(recv)
    if rng.random() < .04:                                          # legitimate collect request
        tt, ch = "collect_request", "collect"
    amt = max(20, a.average_transaction_amount * float(np.exp(nrng.normal(0, .45))))
    add(s, recv, rand_time(), amt, ttype=tt, channel=ch, new=int(is_new),
        device=f"DEV_N{rng.randint(1000, 9999)}" if rng.random() < .03 else None,
        loc=other_loc(s) if rng.random() < .12 else None)

# ----------------------------------------------------------- finalise ledger
df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.insert(0, "transaction_id", [f"TXN_{i:05d}" for i in range(1, len(df) + 1)])
df["timestamp"] = pd.to_datetime(df["timestamp"])
v1, v24 = np.zeros(len(df), int), np.zeros(len(df), int)
for _, g in df.groupby("sender_id"):                                # counts in (t-1h, t] and (t-24h, t]
    t = g["timestamp"].values.astype("datetime64[s]").astype("int64")
    right = np.searchsorted(t, t, side="right")
    v1[g.index] = right - np.searchsorted(t, t - 3600, side="right")
    v24[g.index] = right - np.searchsorted(t, t - 86400, side="right")
df["txn_velocity_1h"], df["txn_velocity_24h"] = v1, v24
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
df = df[["transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "transaction_type", "payment_channel",
         "device_id", "location_id", "sender_balance", "account_age_days", "beneficiary_age_days",
         "txn_velocity_1h", "txn_velocity_24h", "is_new_beneficiary", "is_fraud", "fraud_type", "scenario_id"]]

fr = df[df.is_fraud == 1]
cases = fr.groupby("scenario_id").agg(
    fraud_type=("fraud_type", "first"), n_transactions=("transaction_id", "count"), total_amount=("amount", "sum"),
    first_timestamp=("timestamp", "min"), sender_ids=("sender_id", lambda x: ";".join(sorted(set(x)))),
    beneficiary_ids=("receiver_id", lambda x: ";".join(sorted(set(x)))),
    device_ids=("device_id", lambda x: ";".join(sorted(set(x))))).reset_index()
cases.insert(0, "case_id", [f"CASE_{i:03d}" for i in range(1, len(cases) + 1)])

accounts.to_csv(OUT / "accounts.csv", index=False)
ben.to_csv(OUT / "beneficiaries.csv", index=False)
df.to_csv(OUT / "transactions.csv", index=False)
cases.to_csv(OUT / "fraud_cases.csv", index=False)

indeg = df.groupby("receiver_id")["sender_id"].nunique().sort_values(ascending=False)
shared = accounts.groupby("device_id").account_id.nunique()
print(f"transactions: {len(df)} | accounts: {len(accounts)} | beneficiaries: {len(ben)} | cases: {len(cases)}")
print("\nFraud distribution:\n", df.fraud_type.value_counts().to_string())
print("\nHard negatives (legit):", df.scenario_id.isin(["LEGIT_LARGE", "LEGIT_BURST"]).sum())
print("Max 1h velocity in rapid bursts:", int(df[df.fraud_type == "rapid_transfer"].txn_velocity_1h.max()),
      "| in legit bursts:", int(df[df.scenario_id == "LEGIT_BURST"].txn_velocity_1h.max()))
print("\nTop beneficiaries by unique senders (legit hubs vs mules):")
for b_, n in indeg.head(8).items():
    print(f"  {b_}  senders={n:<3} {'MULE' if b_ in mules else ben_i.loc[b_, 'category']}")
print("\nDevices used by >1 account:", int((shared > 1).sum()), "(4 are fraud rings, 15 are legit family pairs)")