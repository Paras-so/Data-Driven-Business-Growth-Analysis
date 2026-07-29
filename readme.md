A Python based end to end business growth analytics pipeline that ingests and processes 24 months of real company data, including monthly revenue transactions peaking at ₹181,577 in Year 2, customer acquisition records tracking 48,959 new customers, churn logs recording 6,518 lost customers, average order values of ₹50, and multi-stage marketing funnel data spanning 126,008 monthly visitors down to 1,783 final conversions, to perform in depth analysis across three core business domains: Revenue & Sales, Customer Acquisition & Retention, and Marketing & Funnel Metrics. The system automatically computes critical KPIs such as 31.4% Year over Year revenue growth, a Compound Annual Growth Rate (CAGR) of 31.4%, Customer Lifetime Value (LTV) of ₹1,923, Customer Acquisition Cost (CAC) of ₹48, an exceptional LTV:CAC ratio of 41.57×, a 95.1% average monthly retention rate, and a 9.31% visitor to lead conversion rate across a 126,008 strong monthly visitor base. Results are delivered through four multi panel executive level dashboards built with Matplotlib and a structured plain-text KPI report, giving business stakeholders a complete, data driven picture of company performance and actionable insights for strategic growth decisions.# Data-Driven Business Growth Analysis

An end-to-end Python analytics pipeline that processes 24 months of company data to surface actionable insights across revenue, customer behavior, and marketing performance.

## Overview

This project ingests and analyzes real business data to answer three core questions: how is revenue growing, how well are we acquiring and retaining customers, and how effectively is our marketing funnel converting visitors into customers.

## Dataset

- **Revenue transactions:** 24 months, peaking at ₹181,577 in Year 2
- **Customer acquisition:** 48,959 new customers tracked
- **Churn:** 6,518 customers lost over the period
- **Average order value:** ₹50
- **Marketing funnel:** 126,008 monthly visitors → 1,783 final conversions

## Key Metrics Computed

| Metric | Value |
|---|---|
| YoY Revenue Growth | 31.4% |
| CAGR | 31.4% |
| Customer Lifetime Value (LTV) | ₹1,923 |
| Customer Acquisition Cost (CAC) | ₹48 |
| LTV:CAC Ratio | 41.57× |
| Avg. Monthly Retention Rate | 95.1% |
| Visitor-to-Lead Conversion Rate | 9.31% |

## Analysis Domains

1. **Revenue & Sales** — growth trends, YoY performance, CAGR
2. **Customer Acquisition & Retention** — LTV, CAC, retention/churn analysis
3. **Marketing & Funnel Metrics** — visitor-to-conversion funnel breakdown

## Output

- Four multi-panel executive dashboards (Matplotlib):
  - Revenue Overview
  - Customer Metrics
  - Funnel Metrics
  - Executive Summary Dashboard
- A structured plain-text KPI report (`growth_analysis_report.txt`)

## Tools & Skills

**Tools:** Python, Pandas, Matplotlib
**Skills:** Data Cleaning, EDA, KPI Computation, Business Analytics, Dashboard Development

## Running It

```bash
python business_growth_analysis.py
```

This generates the dashboard images and KPI report in the project directory.
