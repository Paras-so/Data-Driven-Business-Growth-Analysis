"""
Data-Driven Business Growth Analysis
=====================================
Covers:
  - Revenue & Sales Growth
  - Customer Acquisition & Retention
  - Marketing & Funnel Metrics

Generates synthetic 2-year monthly data, runs analysis,
prints a summary report, and saves charts to /outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import date
import warnings
import os

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Style ─────────────────────────────────────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════════════
# 1.  DATA GENERATION
# ════════════════════════════════════════════════════════════════════════════════

N_MONTHS = 24   # 2 years
months   = pd.date_range("2023-01-01", periods=N_MONTHS, freq="MS")

# --- Revenue & Sales ---------------------------------------------------------
base_rev  = 120_000
trend     = np.linspace(0, 80_000, N_MONTHS)
seasonal  = 15_000 * np.sin(np.linspace(0, 4 * np.pi, N_MONTHS))
noise_rev = rng.normal(0, 6_000, N_MONTHS)
revenue   = np.maximum(base_rev + trend + seasonal + noise_rev, 0)

units_sold  = (revenue / rng.uniform(45, 55, N_MONTHS)).astype(int)
avg_order   = revenue / units_sold

refund_rate = rng.uniform(0.02, 0.06, N_MONTHS)
net_revenue = revenue * (1 - refund_rate)

# --- Customer Metrics --------------------------------------------------------
new_cust   = (units_sold * rng.uniform(0.55, 0.75, N_MONTHS)).astype(int)
churned    = (new_cust * rng.uniform(0.08, 0.18, N_MONTHS)).astype(int)

cust_base = [500]
for i in range(1, N_MONTHS):
    cust_base.append(cust_base[-1] + new_cust[i] - churned[i])
cust_base = np.array(cust_base)

existing_cust   = np.maximum(cust_base - new_cust, 1)
churn_rate      = np.clip(churned / existing_cust, 0, 0.3)
retention_rate  = 1 - churn_rate
cac            = rng.uniform(30, 70, N_MONTHS)       # cost to acquire 1 customer
ltv            = avg_order * (1 / (1 - retention_rate.clip(0, 0.98)))
ltv_cac_ratio  = ltv / cac

# --- Funnel Metrics ----------------------------------------------------------
visitors     = (new_cust / rng.uniform(0.01, 0.025, N_MONTHS)).astype(int)
leads        = (visitors * rng.uniform(0.06, 0.12, N_MONTHS)).astype(int)
mql          = (leads    * rng.uniform(0.35, 0.55, N_MONTHS)).astype(int)
sql          = (mql      * rng.uniform(0.35, 0.55, N_MONTHS)).astype(int)
conversions  = np.minimum(new_cust, sql)

visitor_to_lead  = leads        / np.maximum(visitors, 1) * 100
lead_to_mql      = mql          / np.maximum(leads, 1)    * 100
mql_to_sql       = sql          / np.maximum(mql, 1)      * 100
sql_to_close     = conversions  / np.maximum(sql, 1)      * 100
overall_cvr      = conversions  / np.maximum(visitors, 1) * 100

channels = {
    "Organic Search": rng.uniform(0.28, 0.35, N_MONTHS),
    "Paid Ads":       rng.uniform(0.22, 0.30, N_MONTHS),
    "Email":          rng.uniform(0.15, 0.22, N_MONTHS),
    "Social Media":   rng.uniform(0.10, 0.16, N_MONTHS),
    "Referral":       rng.uniform(0.06, 0.12, N_MONTHS),
}
# normalise so they sum to 1
ch_arr = np.array(list(channels.values()))
ch_arr = ch_arr / ch_arr.sum(axis=0, keepdims=True)
for i, k in enumerate(channels):
    channels[k] = ch_arr[i]

# ── Master DataFrame ──────────────────────────────────────────────────────────
df = pd.DataFrame({
    "month":            months,
    "revenue":          revenue,
    "net_revenue":      net_revenue,
    "units_sold":       units_sold,
    "avg_order_value":  avg_order,
    "refund_rate":      refund_rate,
    "new_customers":    new_cust,
    "churned":          churned,
    "total_customers":  cust_base,
    "retention_rate":   retention_rate,
    "cac":              cac,
    "ltv":              ltv,
    "ltv_cac_ratio":    ltv_cac_ratio,
    "visitors":         visitors,
    "leads":            leads,
    "mql":              mql,
    "sql":              sql,
    "conversions":      conversions,
    "visitor_to_lead":  visitor_to_lead,
    "lead_to_mql":      lead_to_mql,
    "mql_to_sql":       mql_to_sql,
    "sql_to_close":     sql_to_close,
    "overall_cvr":      overall_cvr,
    **{f"ch_{k.lower().replace(' ','_')}": v for k, v in channels.items()},
})

yr1 = df.iloc[:12]
yr2 = df.iloc[12:]


# ════════════════════════════════════════════════════════════════════════════════
# 2.  ANALYSIS HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def yoy(col):
    """Year-over-year growth %."""
    return (yr2[col].sum() - yr1[col].sum()) / yr1[col].sum() * 100

def mom_growth(series):
    """Month-over-month growth % series."""
    return series.pct_change() * 100

def cagr(start, end, periods):
    return ((end / start) ** (1 / periods) - 1) * 100


# ════════════════════════════════════════════════════════════════════════════════
# 3.  CHARTS
# ════════════════════════════════════════════════════════════════════════════════

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


# ── Chart 1: Revenue Overview ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Revenue & Sales Growth", fontsize=15, fontweight="bold", y=1.01)

ax = axes[0, 0]
ax.fill_between(df.month, df.revenue / 1000, alpha=0.15, color=PALETTE[0])
ax.plot(df.month, df.revenue / 1000, color=PALETTE[0], lw=2, label="Gross")
ax.plot(df.month, df.net_revenue / 1000, color=PALETTE[1], lw=1.5, ls="--", label="Net")
ax.axvline(df.month.iloc[12], color=GRID_CLR, lw=1.2)
ax.set_title("Monthly Revenue (₹ thousands)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
ax.legend(fontsize=9)
ax.grid(axis="y")

ax = axes[0, 1]
mom = mom_growth(df.revenue).fillna(0)
colors = [PALETTE[2] if v >= 0 else PALETTE[1] for v in mom]
ax.bar(df.month, mom, color=colors, width=20)
ax.axhline(0, color=TEXT_CLR, lw=0.8)
ax.set_title("Month-over-Month Revenue Growth (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.grid(axis="y")

ax = axes[1, 0]
ax.bar(df.month, df.units_sold, color=PALETTE[3], alpha=0.8, width=20)
ax.set_title("Units Sold")
ax.grid(axis="y")

ax = axes[1, 1]
ax.plot(df.month, df.avg_order_value, color=PALETTE[4], lw=2, marker="o", ms=4)
ax.set_title("Average Order Value (₹)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}"))
ax.grid(axis="y")

fig.tight_layout()
save(fig, "01_revenue_overview.png")


# ── Chart 2: Customer Metrics ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Customer Acquisition & Retention", fontsize=15, fontweight="bold", y=1.01)

ax = axes[0, 0]
ax.fill_between(df.month, df.total_customers, alpha=0.12, color=PALETTE[0])
ax.plot(df.month, df.total_customers, color=PALETTE[0], lw=2.5)
ax2 = ax.twinx()
ax2.bar(df.month, df.new_customers, color=PALETTE[2], alpha=0.45, width=20, label="New")
ax2.bar(df.month, -df.churned, color=PALETTE[1], alpha=0.45, width=20, label="Churned")
ax2.axhline(0, color=TEXT_CLR, lw=0.6)
ax2.spines["top"].set_visible(False)
ax2.tick_params(colors=TEXT_CLR)
ax2.legend(fontsize=9, loc="upper left")
ax.set_title("Customer Base Growth")
ax.set_ylabel("Total customers")
ax.grid(axis="y")

ax = axes[0, 1]
ax.plot(df.month, df.retention_rate * 100, color=PALETTE[2], lw=2)
ax.fill_between(df.month, df.retention_rate * 100, 80, alpha=0.07, color=PALETTE[2])
ax.axhline(90, color=PALETTE[1], lw=1, ls="--", label="90% benchmark")
ax.set_title("Monthly Retention Rate (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylim(75, 100)
ax.legend(fontsize=9)
ax.grid(axis="y")

ax = axes[1, 0]
ax.plot(df.month, df.cac, color=PALETTE[1], lw=2, label="CAC")
ax.plot(df.month, df.ltv, color=PALETTE[0], lw=2, label="LTV")
ax.set_title("CAC vs LTV (₹)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.0f}"))
ax.legend(fontsize=9)
ax.grid(axis="y")

ax = axes[1, 1]
colors_ratio = [PALETTE[2] if v >= 3 else PALETTE[4] if v >= 2 else PALETTE[1]
                for v in df.ltv_cac_ratio]
ax.bar(df.month, df.ltv_cac_ratio, color=colors_ratio, width=20)
ax.axhline(3, color=PALETTE[2], lw=1, ls="--", label="3× (healthy)")
ax.axhline(2, color=PALETTE[4], lw=1, ls="--", label="2× (minimum)")
ax.set_title("LTV : CAC Ratio")
ax.legend(fontsize=9)
ax.grid(axis="y")

fig.tight_layout()
save(fig, "02_customer_metrics.png")


# ── Chart 3: Marketing Funnel ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Marketing & Funnel Metrics", fontsize=15, fontweight="bold", y=1.01)

# Funnel bars (averages)
ax = axes[0]
stages = ["Visitors", "Leads", "MQLs", "SQLs", "Conversions"]
avg_vals = [df.visitors.mean(), df.leads.mean(),
            df.mql.mean(),   df.sql.mean(), df.conversions.mean()]
bar_colors = [PALETTE[0], PALETTE[2], PALETTE[3], PALETTE[4], PALETTE[1]]
bars = ax.barh(stages[::-1], avg_vals[::-1], color=bar_colors[::-1], height=0.55)
for bar, val in zip(bars, avg_vals[::-1]):
    ax.text(bar.get_width() + max(avg_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", fontsize=9)
ax.set_title("Avg Monthly Funnel Volume")
ax.set_xlabel("Count")
ax.grid(axis="x")
ax.set_xlim(0, max(avg_vals) * 1.18)

# Conversion rates over time
ax = axes[1]
ax.plot(df.month, df.visitor_to_lead, color=PALETTE[0], lw=1.6, label="Visitor→Lead")
ax.plot(df.month, df.mql_to_sql,      color=PALETTE[2], lw=1.6, label="MQL→SQL")
ax.plot(df.month, df.sql_to_close,    color=PALETTE[1], lw=1.6, label="SQL→Close")
ax.plot(df.month, df.overall_cvr,     color=PALETTE[5], lw=1.2, ls="--", label="Overall CVR")
ax.set_title("Conversion Rates (%)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.legend(fontsize=8)
ax.grid(axis="y")

# Channel mix (last month)
ax = axes[2]
ch_labels = ["Organic\nSearch", "Paid\nAds", "Email", "Social\nMedia", "Referral"]
ch_vals   = [df[f"ch_{k.lower().replace(' ','_')}"].iloc[-1] * 100
             for k in channels]
wedge_colors = PALETTE[:5]
wedges, texts, autotexts = ax.pie(
    ch_vals, labels=ch_labels, autopct="%1.1f%%",
    colors=wedge_colors, startangle=90,
    pctdistance=0.78, textprops={"fontsize": 8},
    wedgeprops={"linewidth": 1.2, "edgecolor": BG}
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color("white")
ax.set_title("Traffic Channel Mix\n(latest month)")

fig.tight_layout()
save(fig, "03_funnel_metrics.png")


# ── Chart 4: Executive KPI Dashboard ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.suptitle("Executive KPI Summary  ·  2023–2024", fontsize=14, fontweight="bold")

kpis = [
    ("Total Revenue",        f"₹{df.revenue.sum()/1e6:.2f}M",  f"YoY +{yoy('revenue'):.1f}%",          PALETTE[0]),
    ("Total Customers",      f"{df.total_customers.iloc[-1]:,}", f"Started at {df.total_customers.iloc[0]:,}", PALETTE[2]),
    ("Avg Retention Rate",   f"{df.retention_rate.mean()*100:.1f}%", f"Monthly avg",                    PALETTE[3]),
    ("Avg LTV : CAC",        f"{df.ltv_cac_ratio.mean():.2f}×",  "3× = healthy",                       PALETTE[4]),
    ("Overall Conv. Rate",   f"{df.overall_cvr.mean():.2f}%",    "Visitor→Sale",                        PALETTE[1]),
    ("Revenue CAGR",         f"{cagr(yr1.revenue.mean(), yr2.revenue.mean(), 1):.1f}%",  "Annualised",  PALETTE[5]),
]

gs = fig.add_gridspec(1, 6, wspace=0.35)
for i, (label, val, sub, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor=color + "18", edgecolor=color + "55",
                                linewidth=1.2, clip_on=False,
                                zorder=0, joinstyle="round"))
    ax.text(0.5, 0.72, label, ha="center", va="center", fontsize=9,
            color=TEXT_CLR, alpha=0.7, transform=ax.transAxes)
    ax.text(0.5, 0.45, val, ha="center", va="center", fontsize=16,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=8,
            color=TEXT_CLR, alpha=0.5, transform=ax.transAxes)

fig.tight_layout()
save(fig, "04_executive_dashboard.png")


# ════════════════════════════════════════════════════════════════════════════════
# 4.  TEXT SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════════

div = "=" * 64
sep = "-" * 64

report = f"""
{div}
  DATA-DRIVEN BUSINESS GROWTH ANALYSIS
  Period: Jan 2023 – Dec 2024  |  Generated: {date.today()}
{div}

{'─── REVENUE & SALES ───':^64}

  Gross Revenue (2-yr total)   ₹{df.revenue.sum():>12,.0f}
  Net Revenue   (2-yr total)   ₹{df.net_revenue.sum():>12,.0f}
  YoY Revenue Growth           {yoy('revenue'):>+10.1f}%
  Revenue CAGR (annualised)    {cagr(yr1.revenue.mean(), yr2.revenue.mean(), 1):>+10.1f}%
  Avg Monthly Revenue Yr1      ₹{yr1.revenue.mean():>12,.0f}
  Avg Monthly Revenue Yr2      ₹{yr2.revenue.mean():>12,.0f}
  Total Units Sold             {df.units_sold.sum():>12,}
  Avg Order Value (all-time)   ₹{df.avg_order_value.mean():>12,.0f}
  Avg Refund Rate              {df.refund_rate.mean()*100:>11.1f}%
  Peak Revenue Month           {df.loc[df.revenue.idxmax(), 'month'].strftime('%b %Y')}
  Best MoM Growth              {mom_growth(df.revenue).max():>+10.1f}%

{sep}
{'─── CUSTOMER ACQUISITION & RETENTION ───':^64}

  Total Customers (end)        {int(df.total_customers.iloc[-1]):>12,}
  Total New Customers Acq.     {int(df.new_customers.sum()):>12,}
  Total Churned                {int(df.churned.sum()):>12,}
  YoY New-Customer Growth      {yoy('new_customers'):>+10.1f}%
  Avg Monthly Retention Rate   {df.retention_rate.mean()*100:>11.1f}%
  Avg CAC                      ₹{df.cac.mean():>12,.0f}
  Avg LTV                      ₹{df.ltv.mean():>12,.0f}
  Avg LTV : CAC Ratio          {df.ltv_cac_ratio.mean():>12.2f}×
    → Healthy threshold = 3×; Warning < 2×

{sep}
{'─── MARKETING & FUNNEL ───':^64}

  Avg Monthly Visitors         {df.visitors.mean():>12,.0f}
  Avg Monthly Leads            {df.leads.mean():>12,.0f}
  Avg Monthly MQLs             {df.mql.mean():>12,.0f}
  Avg Monthly SQLs             {df.sql.mean():>12,.0f}
  Avg Monthly Conversions      {df.conversions.mean():>12,.0f}

  Visitor → Lead CVR           {df.visitor_to_lead.mean():>11.2f}%
  Lead   → MQL   CVR           {df.lead_to_mql.mean():>11.2f}%
  MQL    → SQL   CVR           {df.mql_to_sql.mean():>11.2f}%
  SQL    → Close CVR           {df.sql_to_close.mean():>11.2f}%
  Overall Visitor → Sale CVR   {df.overall_cvr.mean():>11.2f}%

  Top Traffic Channel (latest) Organic Search

{sep}
{'─── KEY INSIGHTS ───':^64}

  1. Revenue grew {yoy('revenue'):.1f}% YoY — driven by both volume
     and AOV improvement over the 24-month window.

  2. Retention averaged {df.retention_rate.mean()*100:.1f}%, keeping churn manageable.
     Months below 88% warrant investigation.

  3. LTV:CAC ratio of {df.ltv_cac_ratio.mean():.2f}× is {'above' if df.ltv_cac_ratio.mean()>=3 else 'approaching'} the 3× healthy
     benchmark — unit economics are {'solid' if df.ltv_cac_ratio.mean()>=3 else 'improving'}.

  4. Funnel top-of-funnel CVR ({df.visitor_to_lead.mean():.1f}%) is the biggest
     leverage point — improving it by 1pp compounds across
     every downstream stage.

  5. Organic Search dominates traffic mix; diversifying
     paid and email channels can reduce concentration risk.

{div}
  Charts saved:
    01_revenue_overview.png
    02_customer_metrics.png
    03_funnel_metrics.png
    04_executive_dashboard.png
{div}
"""

print(report)

# Save report as text file
report_path = os.path.join(OUT, "growth_analysis_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(f"\n  All outputs saved to: {OUT}\n")
