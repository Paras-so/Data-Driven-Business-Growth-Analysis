# Business Growth Analysis - 2 years of data (2023-2024)
# covers revenue, customers, and marketing funnel
# extended with a PostgreSQL-backed NL2SQL query layer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import date
import warnings
import os

warnings.filterwarnings("ignore")

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

PALETTE   = ["#3266AD", "#D85A30", "#2E9E75", "#8B5CF6", "#F59E0B", "#64748B"]
BG        = "#FAFAF9"
GRID_CLR  = "#E5E3DC"
TEXT_CLR  = "#1C1C1A"
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "text.color":       TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "grid.color":       GRID_CLR,
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

N_MONTHS = 24
months   = pd.date_range("2023-01-01", periods=N_MONTHS, freq="MS")

base_rev  = 120_000
trend     = np.linspace(0, 80_000, N_MONTHS)
seasonal  = 15_000 * np.sin(np.linspace(0, 4 * np.pi, N_MONTHS))
noise_rev = rng.normal(0, 6_000, N_MONTHS)
revenue   = np.maximum(base_rev + trend + seasonal + noise_rev, 0)

units_sold = (revenue / rng.uniform(45, 55, N_MONTHS)).astype(int)
avg_order  = revenue / units_sold

refund_rate = rng.uniform(0.02, 0.06, N_MONTHS)
net_revenue = revenue * (1 - refund_rate)

new_cust = (units_sold * rng.uniform(0.55, 0.75, N_MONTHS)).astype(int)
churned  = (new_cust * rng.uniform(0.08, 0.18, N_MONTHS)).astype(int)

cust_base = [500]
for i in range(1, N_MONTHS):
    cust_base.append(cust_base[-1] + new_cust[i] - churned[i])
cust_base = np.array(cust_base)

existing_cust  = np.maximum(cust_base - new_cust, 1)
churn_rate     = np.clip(churned / existing_cust, 0, 0.3)
retention_rate = 1 - churn_rate

cac = rng.uniform(30, 70, N_MONTHS)
ltv = avg_order * (1 / (1 - retention_rate.clip(0, 0.98)))
ltv_cac_ratio = ltv / cac

visitors    = (new_cust / rng.uniform(0.01, 0.025, N_MONTHS)).astype(int)
leads       = (visitors * rng.uniform(0.06, 0.12, N_MONTHS)).astype(int)
mql         = (leads * rng.uniform(0.35, 0.55, N_MONTHS)).astype(int)
sql         = (mql * rng.uniform(0.35, 0.55, N_MONTHS)).astype(int)
conversions = np.minimum(new_cust, sql)

visitor_to_lead = leads / np.maximum(visitors, 1) * 100
lead_to_mql     = mql / np.maximum(leads, 1) * 100
mql_to_sql      = sql / np.maximum(mql, 1) * 100
sql_to_close    = conversions / np.maximum(sql, 1) * 100
overall_cvr     = conversions / np.maximum(visitors, 1) * 100

channels = {
    "Organic Search": rng.uniform(0.28, 0.35, N_MONTHS),
    "Paid Ads":       rng.uniform(0.22, 0.30, N_MONTHS),
    "Email":          rng.uniform(0.15, 0.22, N_MONTHS),
    "Social Media":   rng.uniform(0.10, 0.16, N_MONTHS),
    "Referral":       rng.uniform(0.06, 0.12, N_MONTHS),
}

ch_arr = np.array(list(channels.values()))
ch_arr = ch_arr / ch_arr.sum(axis=0, keepdims=True)
for i, k in enumerate(channels):
    channels[k] = ch_arr[i]

df = pd.DataFrame({
    "month": months,
    "revenue": revenue,
    "net_revenue": net_revenue,
    "units_sold": units_sold,
    "avg_order_value": avg_order,
    "refund_rate": refund_rate,
    "new_customers": new_cust,
    "churned": churned,
    "total_customers": cust_base,
    "retention_rate": retention_rate,
    "cac": cac,
    "ltv": ltv,
    "ltv_cac_ratio": ltv_cac_ratio,
    "visitors": visitors,
    "leads": leads,
    "mql": mql,
    "sql": sql,
    "conversions": conversions,
    "visitor_to_lead": visitor_to_lead,
    "lead_to_mql": lead_to_mql,
    "mql_to_sql": mql_to_sql,
    "sql_to_close": sql_to_close,
    "overall_cvr": overall_cvr,
    **{f"ch_{k.lower().replace(' ','_')}": v for k, v in channels.items()},
})

yr1 = df.iloc[:12]
yr2 = df.iloc[12:]

def yoy(col):
    return (yr2[col].sum() - yr1[col].sum()) / yr1[col].sum() * 100

def mom_growth(series):
    return series.pct_change() * 100

def cagr(start, end, periods):
    return ((end / start) ** (1 / periods) - 1) * 100

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path

# ================================================================
# NL2SQL Retail Insights Layer
# ================================================================

from sqlalchemy import create_engine, text

DB_USER = "postgres"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "business_growth"

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def load_to_db(df, table_name="transactions", db_url=DB_URL):
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    engine.dispose()
    return db_url


def get_schema(table_name="transactions", db_url=DB_URL):
    engine = create_engine(db_url)
    query = text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position
    """)
    with engine.connect() as conn:
        columns = conn.execute(query, {"table_name": table_name}).fetchall()
    engine.dispose()

    schema_desc = f"Table: {table_name}\nColumns:\n"
    for col in columns:
        schema_desc += f"  - {col[0]} ({col[1]})\n"
    return schema_desc


def is_safe_query(sql):
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "grant"]
    sql_lower = sql.strip().lower()
    if not sql_lower.startswith("select"):
        return False
    return not any(word in sql_lower for word in forbidden)


def nl_to_sql(question, schema_desc, client):
    prompt = f"""Given this PostgreSQL schema:

{schema_desc}

Write a single valid PostgreSQL SELECT query to answer this question:
"{question}"

Return ONLY the SQL query, no explanation."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.content[0].text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def ask(question, table_name="transactions", db_url=DB_URL, client=None):
    schema_desc = get_schema(table_name, db_url)
    sql = nl_to_sql(question, schema_desc, client)

    if not is_safe_query(sql):
        return {"error": "Unsafe query blocked", "sql": sql}

    engine = create_engine(db_url)
    try:
        result = pd.read_sql_query(sql, engine)
    except Exception as e:
        return {"error": str(e), "sql": sql}
    finally:
        engine.dispose()

    return {"sql": sql, "result": result}


