# Cohort Analysis with RLM + DataFrame[¶](#cohort-analysis-with-rlm-dataframe)

This notebook demonstrates how to pass large DataFrames to a DSPy RLM (Recursive Language Model) for iterative data analysis.

We'll give the RLM three DataFrames from a SaaS product — users, events, and subscriptions — and ask it to investigate why retention is dropping. The RLM will iteratively explore the data, writing and executing Python code in a sandbox, until it identifies the root cause.

**Expected findings:**

- Overall churn rate: ~23-25%
- Worst channel: `paid_campaign_x` (~45% churn vs ~15% organic)
- Key insight: churned users from `paid_campaign_x` never use `advanced_reports` — early usage of that feature is an activation signal

**Prerequisites:**

1. Run `generate_cohort_data.py` to create the parquet files
1. Set your LLM API key (e.g. `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)

## 1. Generate the dataset[¶](#1-generate-the-dataset)

Run `generate_cohort_data.py` to create three parquet files with embedded bias signals. You only need to do this once.

In \[1\]:

Copied!

```
!python generate_cohort_data.py
```

!python generate_cohort_data.py

```
Generating users...
Generating subscriptions...
Generating events...

Dataset stats:
  Users:             10,000 rows
  Subscriptions:     10,000 rows
  Events:           250,918 rows
  Total memory:        16.3 MB

Embedded signal:
  paid_campaign_x churn rate: 45%
  organic churn rate:         16%

Saved to: users.parquet, subscriptions.parquet, events.parquet
```

## 2. Imports and setup[¶](#2-imports-and-setup)

In \[2\]:

Copied!

```
import pandas as pd

import dspy

# Import the standalone DataFrame wrapper (implements SandboxSerializable)
from dataframe import DataFrame
```

import pandas as pd import dspy

# Import the standalone DataFrame wrapper (implements SandboxSerializable)

from dataframe import DataFrame

## 3. Load and inspect the data[¶](#3-load-and-inspect-the-data)

In \[3\]:

Copied!

```
users = pd.read_parquet("users.parquet")
events = pd.read_parquet("events.parquet")
subscriptions = pd.read_parquet("subscriptions.parquet")

print(f"Users:         {len(users):,} rows")
print(f"Events:        {len(events):,} rows")
print(f"Subscriptions: {len(subscriptions):,} rows")
```

users = pd.read_parquet("users.parquet") events = pd.read_parquet("events.parquet") subscriptions = pd.read_parquet("subscriptions.parquet") print(f"Users: {len(users):,} rows") print(f"Events: {len(events):,} rows") print(f"Subscriptions: {len(subscriptions):,} rows")

```
Users:         10,000 rows
Events:        250,918 rows
Subscriptions: 10,000 rows
```

In \[4\]:

Copied!

```
users.head()
```

users.head()

Out\[4\]:

|     | user_id | email                     | name                | signup_date | acquisition_channel | plan_at_signup | country |
| --- | ------- | ------------------------- | ------------------- | ----------- | ------------------- | -------------- | ------- |
| 0   | 1       | johnsonjoshua@example.org | Brian Yang          | 2024-01-29  | organic             | free           | US      |
| 1   | 2       | garzaanthony@example.org  | Jonathan Johnson    | 2024-01-27  | paid_campaign_x     | pro            | US      |
| 2   | 3       | jennifermiles@example.com | Kevin Pacheco       | 2024-04-18  | organic             | free           | US      |
| 3   | 4       | blakeerik@example.com     | Christopher Bernard | 2024-01-07  | paid_campaign_x     | pro            | AU      |
| 4   | 5       | curtis61@example.com      | Lindsey Roman       | 2024-04-17  | organic             | starter        | BR      |

In \[5\]:

Copied!

```
subscriptions.head()
```

subscriptions.head()

Out\[5\]:

|     | user_id | subscription_start | subscription_end | plan    | mrr | status    | cancellation_reason |
| --- | ------- | ------------------ | ---------------- | ------- | --- | --------- | ------------------- |
| 0   | 1       | 2024-01-29         | NaT              | free    | 0   | active    | NaN                 |
| 1   | 2       | 2024-01-27         | NaT              | pro     | 79  | active    | NaN                 |
| 2   | 3       | 2024-04-18         | NaT              | free    | 0   | active    | NaN                 |
| 3   | 4       | 2024-01-07         | NaT              | pro     | 79  | active    | NaN                 |
| 4   | 5       | 2024-04-17         | 2024-05-09       | starter | 29  | cancelled | poor_support        |

## 4. Wrap DataFrames for RLM sandbox injection[¶](#4-wrap-dataframes-for-rlm-sandbox-injection)

The `DataFrame` wrapper implements `SandboxSerializable`, which tells the RLM how to:

- **Serialize** the data (base64-encoded Parquet)
- **Reconstruct** it inside the sandbox (`pd.read_parquet(...)`)
- **Preview** it for the LLM (column names, dtypes, sample rows)

The full data is injected once into the sandbox at the start of the session. The LLM sees only a compact preview in its prompt but can write code against the real DataFrame.

In \[6\]:

Copied!

```
wrapped_users = DataFrame(users)
wrapped_events = DataFrame(events)
wrapped_subscriptions = DataFrame(subscriptions)

# This is what the LLM sees in its prompt:
print(wrapped_users.rlm_preview())
```

wrapped_users = DataFrame(users) wrapped_events = DataFrame(events) wrapped_subscriptions = DataFrame(subscriptions)

# This is what the LLM sees in its prompt:

print(wrapped_users.rlm_preview())

```
DataFrame: 10,000 rows x 7 columns

Columns:
  user_id: int64
  email: str
  name: str
  signup_date: datetime64[us]
  acquisition_channel: str
  plan_at_signup: str
  country: str

Sample (first 3 rows):
   user_id                      email              name signup_date acquisition_channel plan_at_signup country
0        1  johnsonjoshua@example.org        Brian Yang  2024-01-29             organic           free      US
1        2   garzaanthony@example.org  Jonathan Johnson  2024-01-27     p...
```

## 5. Define the signature[¶](#5-define-the-signature)

The signature declares three `DataFrame` inputs and four structured outputs. The docstring becomes the system prompt for the RLM.

In \[7\]:

Copied!

```
class CohortRetentionAnalysis(dspy.Signature):
    """You are a data analyst investigating why user retention is dropping.

    You have access to three DataFrames:
    - `users`: user profiles with signup dates and acquisition channels
    - `events`: feature usage events with timestamps
    - `subscriptions`: subscription status, plan, MRR, and cancellation reasons

    Investigate the data step by step. Compute retention by cohort, segment by
    acquisition channel, compare feature usage between retained and churned users,
    and identify the root cause of churn.
    """

    users: DataFrame = dspy.InputField(desc="User profiles with signup_date, acquisition_channel, country")
    events: DataFrame = dspy.InputField(desc="Feature usage events with user_id, event_type, timestamp")
    subscriptions: DataFrame = dspy.InputField(desc="Subscription records with status, plan, mrr, cancellation_reason")

    overall_churn_rate: float = dspy.OutputField(desc="Overall churn rate as a decimal (e.g. 0.25 for 25%)")
    worst_channel: str = dspy.OutputField(desc="Acquisition channel with highest churn rate")
    key_finding: str = dspy.OutputField(desc="The main insight about what differentiates churned users")
    recommendations: str = dspy.OutputField(desc="2-3 actionable recommendations based on the analysis")
```

class CohortRetentionAnalysis(dspy.Signature): """You are a data analyst investigating why user retention is dropping. You have access to three DataFrames:

- `users`: user profiles with signup dates and acquisition channels
- `events`: feature usage events with timestamps
- `subscriptions`: subscription status, plan, MRR, and cancellation reasons Investigate the data step by step. Compute retention by cohort, segment by acquisition channel, compare feature usage between retained and churned users, and identify the root cause of churn. """ users: DataFrame = dspy.InputField(desc="User profiles with signup_date, acquisition_channel, country") events: DataFrame = dspy.InputField(desc="Feature usage events with user_id, event_type, timestamp") subscriptions: DataFrame = dspy.InputField(desc="Subscription records with status, plan, mrr, cancellation_reason") overall_churn_rate: float = dspy.OutputField(desc="Overall churn rate as a decimal (e.g. 0.25 for 25%)") worst_channel: str = dspy.OutputField(desc="Acquisition channel with highest churn rate") key_finding: str = dspy.OutputField(desc="The main insight about what differentiates churned users") recommendations: str = dspy.OutputField(desc="2-3 actionable recommendations based on the analysis")

## 6. Configure the LM and run[¶](#6-configure-the-lm-and-run)

Update the model name below to match your provider. The RLM will run up to 15 iterations, writing and executing Python code in a sandboxed interpreter at each step.

In \[8\]:

Copied!

```
lm = dspy.LM("openai/gpt-5.4")  # or "anthropic/claude-sonnet-4-5", etc.
dspy.configure(lm=lm)
```

lm = dspy.LM("openai/gpt-5.4") # or "anthropic/claude-sonnet-4-5", etc. dspy.configure(lm=lm)

In \[9\]:

Copied!

```
analyzer = dspy.RLM(
    CohortRetentionAnalysis,
    max_iters=15,
    verbose=True,
)

result = analyzer(
    users=wrapped_users,
    events=wrapped_events,
    subscriptions=wrapped_subscriptions,
)
```

analyzer = dspy.RLM( CohortRetentionAnalysis, max_iters=15, verbose=True, ) result = analyzer( users=wrapped_users, events=wrapped_events, subscriptions=wrapped_subscriptions, )

````
/Users/kmad/dev/dspy-fresh/.venv/lib/python3.13/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:
  PydanticSerializationUnexpectedValue(Expected 9 fields but got 6: Expected `Message` - serialized value may not be as expected [field_name='choices', input_value=Message(content='[[ ## re...: None}, annotations=[]), input_type=Message])
  PydanticSerializationUnexpectedValue(Expected `StreamingChoices` - serialized value may not be as expected [field_name='choices', input_value=Choices(finish_reason='st... None}, annotations=[])), input_type=Choices])
  return self.__pydantic_serializer__.to_python(
2026/03/08 15:52:10 INFO dspy.predict.rlm: RLM iteration 1/15
Reasoning: First step is to explore the data structure and key distributions: shapes, date ranges, status values, acquisition channels, plans, nulls, and some sample cancellation reasons/event types. Then I can define churn/retention properly and compute cohort analyses in later steps.
Code:
```python
print("users shape:", users.shape)
print("events shape:", events.shape)
print("subscriptions shape:", subscriptions.shape)

print("\nusers dtypes:\n", users.dtypes)
print("\nsubscriptions dtypes:\n", subscriptions.dtypes)
print("\nevents dtypes:\n", events.dtypes)

print("\nusers head:\n", users.head(3))
print("\nsubscriptions head:\n", subscriptions.head(3))
print("\nevents head:\n", events.head(3))

print("\nDate ranges:")
print("users signup_date:", users["signup_date"].min(), "to", users["signup_date"].max())
print("subscriptions start:", subscriptions["subscription_start"].min(), "to", subscriptions["subscription_start"].max())
print("subscriptions end:", subscriptions["subscription_end"].min(), "to", subscriptions["subscription_end"].max())
print("events timestamp:", events["timestamp"].min(), "to", events["timestamp"].max())

print("\nAcquisition channels:\n", users["acquisition_channel"].value_counts(dropna=False))
print("\nPlan at signup:\n", users["plan_at_signup"].value_counts(dropna=False))
print("\nSubscription status:\n", subscriptions["status"].value_counts(dropna=False))
print("\nSubscription plan:\n", subscriptions["plan"].value_counts(dropna=False))
print("\nTop cancellation reasons:\n", subscriptions["cancellation_reason"].value_counts(dropna=False).head(15))
print("\nTop event types:\n", events["event_type"].value_counts().head(20))
````

2026/03/08 15:52:10 INFO dspy.predict.rlm: Output (3,297 chars): users shape: (10000, 7) events shape: (250918, 5) subscriptions shape: (10000, 7)

users dtypes: user_id int64 email object name object signup_date datetime64[us] acquisition_channel object plan_at_signup object country object dtype: object

subscriptions dtypes: user_id int64 subscription_start datetime64[us] subscription_end datetime64[us] plan object mrr int64 status object cancellation_reason object dtype: object

events dtypes: user_id int64 event_type object timestamp datetime64[us] session_id object duration_seconds int64 dtype: object

users head: user_id email ... plan_at_signup country 0 1 johnsonjoshua@example.org ... free US 1 2 garzaanthony@example.org ... pro US 2 3 jennifermiles@example.com ... free US

[3 rows x 7 columns]

subscriptions head: user_id subscription_start subscription_end ... mrr status cancellation_reason 0 1 2024-01-29 NaT ... 0 active None 1 2 2024-01-27 NaT ... 79 active None 2 3 2024-04-18 NaT ... 0 active None

[3 rows x 7 columns]

events head: user_id event_type timestamp session_id duration_seconds 0 9910 report_export 2024-01-01 02:22:19 97b17398 263 1 4854 file_upload 2024-01-01 04:24:55 cda2b255 28 2 2520 search 2024-01-01 04:34:43 f833547a 183

Date ranges: users signup_date: 2024-01-01 00:00:00 to 2024-05-30 00:00:00 subscriptions start: 2024-01-01 00:00:00 to 2024-05-30 00:00:00 subscriptions end: 2024-01-09 00:00:00 to 2024-06-30 00:00:00 events timestamp: 2024-01-01 02:22:19 to 2024-06-29 23:59:11

Acquisition channels: acquisition_channel organic 3030 paid_campaign_x 2469 social 1507 paid_campaign_y 1499 referral 1495 Name: count, dtype: int64

Plan at signup: plan_at_signup free 3936 starter 2977 pro 2074 enterprise 1013 Name: count, dtype: int64

Subscription status: status active 7623 cancelled 2377 Name: count, dtype: int64

Subscription plan: plan free 3936 starter 2977 pro 2074 enterprise 1013 Name: count, dtype: int64

Top cancellation reasons: cancellation_reason None 7623 no_longer_needed 413 switched_competitor 407 poor_support 405 other 403 too_expensive 382 missing_features 367 Name: count, dtype: int64

Top event types: event_type api_call 25527 report_export 25441 search 25345 dashboard_view 25266 integration_setup 25235 settings_update 25184 file_upload 25105 team_invite 25057 notification_click 25052 advanced_reports 23706 Name: count, dtype: int64

2026/03/08 15:52:21 INFO dspy.predict.rlm: RLM iteration 2/15 Reasoning: Next I should define churn/retention precisely and compute the main aggregates. Since `subscriptions.status` already indicates active vs cancelled, overall churn rate can be cancelled / total users. For retention by cohort, monthly signup cohorts with retained = active likely works, but I should also be careful about censoring because later cohorts have less time to churn. A good first pass is to compute churn/retention by signup month and by acquisition channel, plus a cohort-by-channel pivot. Then I can identify the worst channel and inspect whether its churn differs by plan mix. After that I’ll compare feature usage between retained and churned users, ideally normalizing by account age / pre-churn activity windows. First, compute merged base table, churn by cohort/channel/plan, and some cross-tabs. Code:

```python
import pandas as pd
import numpy as np

base = users.merge(subscriptions, on="user_id", how="inner", suffixes=("_user", "_sub"))
base["is_churned"] = base["status"].eq("cancelled")
base["is_retained"] = base["status"].eq("active")
base["signup_month"] = base["signup_date"].dt.to_period("M").astype(str)

overall_churn_rate = base["is_churned"].mean()

print("Overall churn rate:", overall_churn_rate)

channel_stats = (
    base.groupby("acquisition_channel")
    .agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        retained_rate=("is_retained", "mean"),
        avg_mrr=("mrr", "mean")
    )
    .sort_values("churn_rate", ascending=False)
)
print("\nChurn by acquisition channel:\n", channel_stats)

cohort_stats = (
    base.groupby("signup_month")
    .agg(
        users=("user_id", "count"),
        churn_rate=("is_churned", "mean"),
        retained_rate=("is_retained", "mean")
    )
    .sort_index()
)
print("\nChurn by signup cohort:\n", cohort_stats)

cohort_channel = (
    base.groupby(["signup_month", "acquisition_channel"])["is_churned"]
    .mean()
    .unstack()
)
print("\nCohort x channel churn rate:\n", cohort_channel)

plan_channel = (
    base.groupby(["acquisition_channel", "plan"])["is_churned"]
    .agg(["mean", "count"])
    .sort_values(["acquisition_channel", "mean"], ascending=[True, False])
)
print("\nChannel x plan churn:\n", plan_channel)

print("\nOverall churn by plan:\n", base.groupby("plan")["is_churned"].mean().sort_values(ascending=False))
```

2026/03/08 15:52:21 INFO dspy.predict.rlm: Output (2,661 chars): Overall churn rate: 0.2377

Churn by acquisition channel: users churn_rate retained_rate avg_mrr acquisition_channel\
paid_campaign_x 2469 0.447550 0.552450 50.027542 social 1507 0.230922 0.769078 51.277372 organic 3030 0.157756 0.842244 50.890429 referral 1495 0.151839 0.848161 50.537124 paid_campaign_y 1499 0.146097 0.853903 47.946631

Churn by signup cohort: users churn_rate retained_rate signup_month\
2024-01 2030 0.240887 0.759113 2024-02 1828 0.241247 0.758753 2024-03 2115 0.214657 0.785343 2024-04 1991 0.243094 0.756906 2024-05 2036 0.250000 0.750000

Cohort x channel churn rate: acquisition_channel organic paid_campaign_x ... referral social signup_month ...\
2024-01 0.142626 0.456979 ... 0.159509 0.253521 2024-02 0.176580 0.446009 ... 0.163763 0.245487 2024-03 0.143077 0.423301 ... 0.124138 0.190031 2024-04 0.161392 0.425703 ... 0.164875 0.238854 2024-05 0.168634 0.485207 ... 0.146965 0.231511

[5 rows x 5 columns]

Channel x plan churn: mean count acquisition_channel plan\
organic enterprise 0.176101 318 starter 0.159817 876 free 0.155629 1208 pro 0.149682 628 paid_campaign_x free 0.456050 967 enterprise 0.447581 248 starter 0.443700 746 pro 0.437008 508 paid_campaign_y starter 0.174946 463 pro 0.162338 308 enterprise 0.131387 137 free 0.118443 591 referral free 0.181973 588 enterprise 0.141935 155 starter 0.131403 449 pro 0.128713 303 social enterprise 0.245161 155 pro 0.235474 327 free 0.230241 582 starter 0.223476 443

Overall churn by plan: plan enterprise 0.241856 free 0.238821 starter 0.238495 pro 0.232401 Name: is_churned, dtype: float64

2026/03/08 15:52:45 INFO dspy.predict.rlm: RLM iteration 3/15 Reasoning: We’ve established overall churn and clearly identified `paid_campaign_x` as the worst acquisition channel, with materially higher churn across every plan and every cohort. Next, I should investigate feature usage differences between retained and churned users, especially whether campaign_x users engage less with key product features. To avoid bias from post-churn inactivity, compare usage in a standardized early-life window (e.g., first 14 days after signup/start) and also check basic activity metrics like number of events, active days, sessions, and feature adoption rates. Then examine cancellation reasons by channel to see if they align with behavior (e.g., missing features, no longer needed, too expensive). This should help identify a concrete root cause and support recommendations. Code:

```python
import pandas as pd
import numpy as np

# Build event window relative to subscription/user start
base_cols = ["user_id", "signup_date", "acquisition_channel", "plan", "mrr", "status", "is_churned", "is_retained", "cancellation_reason"]
ue = base[base_cols].merge(events, on="user_id", how="left")

ue["days_from_signup"] = (ue["timestamp"] - ue["signup_date"]).dt.days
early = ue[(ue["days_from_signup"] >= 0) & (ue["days_from_signup"] <= 14)].copy()

# User-level early engagement metrics
user_early_metrics = (
    early.groupby("user_id")
    .agg(
        early_events=("event_type", "size"),
        early_active_days=("timestamp", lambda s: s.dt.date.nunique()),
        early_sessions=("session_id", "nunique"),
        early_total_duration=("duration_seconds", "sum"),
        early_unique_features=("event_type", "nunique"),
    )
    .reset_index()
)

# add zeros for users with no events in first 14 days
user_early_metrics = base[base_cols].merge(user_early_metrics, on="user_id", how="left")
for c in ["early_events", "early_active_days", "early_sessions", "early_total_duration", "early_unique_features"]:
    user_early_metrics[c] = user_early_metrics[c].fillna(0)

print("Early engagement by retention status:")
print(
    user_early_metrics.groupby("status")[["early_events","early_active_days","early_sessions","early_total_duration","early_unique_features"]]
    .mean()
)

print("\nEarly engagement by channel and status:")
print(
    user_early_metrics.groupby(["acquisition_channel","status"])[["early_events","early_active_days","early_sessions","early_unique_features"]]
    .mean()
)

# Feature adoption rates in first 14 days
feature_adoption = (
    early.assign(value=1)
    .pivot_table(index="user_id", columns="event_type", values="value", aggfunc="max", fill_value=0)
    .reset_index()
)
feature_adoption = base[base_cols].merge(feature_adoption, on="user_id", how="left").fillna(0)

event_cols = [c for c in feature_adoption.columns if c not in base_cols]
adoption_by_status = feature_adoption.groupby("status")[event_cols].mean().T.sort_values("cancelled")
adoption_by_status["diff_active_minus_cancelled"] = adoption_by_status.get("active", 0) - adoption_by_status.get("cancelled", 0)
print("\nFeature adoption rate in first 14 days by status:")
print(adoption_by_status.sort_values("diff_active_minus_cancelled", ascending=False))

adoption_by_channel_status = feature_adoption.groupby(["acquisition_channel","status"])[event_cols].mean()

print("\nPaid Campaign X early feature adoption vs Organic (all users):")
compare_channels = feature_adoption.groupby("acquisition_channel")[event_cols].mean().loc[["paid_campaign_x","organic"]].T
compare_channels["diff_org_minus_pcx"] = compare_channels["organic"] - compare_channels["paid_campaign_x"]
print(compare_channels.sort_values("diff_org_minus_pcx", ascending=False))

print("\nCancellation reasons by channel (row-normalized):")
cancel_reasons = (
    base[base["is_churned"]]
    .groupby(["acquisition_channel", "cancellation_reason"])
    .size()
    .unstack(fill_value=0)
)
cancel_reasons_pct = cancel_reasons.div(cancel_reasons.sum(axis=1), axis=0)
print(cancel_reasons_pct)

# specifically compare campaign_x churn reasons counts
print("\nCancellation reason counts for paid_campaign_x:")
print(base[(base["acquisition_channel"]=="paid_campaign_x") & (base["is_churned"])]["cancellation_reason"].value_counts())
```

2026/03/08 15:52:47 INFO dspy.predict.rlm: Output (3,543 chars): Early engagement by retention status: early_events ... early_unique_features status ...\
active 4.071232 ... 3.333202 cancelled 4.122423 ... 3.350862

[2 rows x 5 columns]

Early engagement by channel and status: early_events ... early_unique_features acquisition_channel status ...\
organic active 4.116379 ... 3.358934 cancelled 4.175732 ... 3.389121 paid_campaign_x active 4.087243 ... 3.332111 cancelled 4.091403 ... 3.286878 paid_campaign_y active 4.083594 ... 3.342188 cancelled 4.283105 ... 3.593607 referral active 4.022871 ... 3.331230 cancelled 3.911894 ... 3.255507 social active 3.992235 ... 3.270060 cancelled 4.183908 ... 3.410920

[10 rows x 4 columns]

Feature adoption rate in first 14 days by status: status active cancelled diff_active_minus_cancelled advanced_reports 0.341467 0.188894 0.152573 settings_update 0.337269 0.332352 0.004917 api_call 0.332415 0.341186 -0.008771 team_invite 0.329529 0.342448 -0.012919 integration_setup 0.328348 0.342869 -0.014521 notification_click 0.328348 0.349180 -0.020831 search 0.337269 0.362642 -0.025373 dashboard_view 0.338187 0.366008 -0.027821 report_export 0.332546 0.361380 -0.028834 file_upload 0.327824 0.363904 -0.036080

Paid Campaign X early feature adoption vs Organic (all users): acquisition_channel paid_campaign_x organic diff_org_minus_pcx advanced_reports 0.187120 0.351815 0.164695 notification_click 0.333333 0.344554 0.011221 api_call 0.338194 0.338944 0.000750 team_invite 0.335763 0.330033 -0.005730 report_export 0.354800 0.343234 -0.011565 search 0.358040 0.342904 -0.015135 settings_update 0.343459 0.324422 -0.019036 integration_setup 0.346294 0.324422 -0.021872 file_upload 0.354800 0.329373 -0.025427 dashboard_view 0.360065 0.333993 -0.026071

Cancellation reasons by channel (row-normalized): cancellation_reason missing_features ... too_expensive acquisition_channel ...\
organic 0.158996 ... 0.156904 paid_campaign_x 0.149321 ... 0.142986 paid_campaign_y 0.168950 ... 0.173516 referral 0.140969 ... 0.176211 social 0.163793 ... 0.204023

[5 rows x 6 columns]

Cancellation reason counts for paid_campaign_x: cancellation_reason switched_competitor 200 other 197 no_longer_needed 196 poor_support 189 missing_features 165 too_expensive 158 Name: count, dtype: int64

2026/03/08 15:53:03 INFO dspy.predict.rlm: RLM iteration 4/15 Reasoning: We know overall churn is 0.2377 and worst channel appears to be paid_campaign_x with consistently high churn across cohorts and plans. Early-life engagement volume is not lower for churned users, so the issue is likely not simple inactivity. A striking pattern is that retained users have much higher advanced_reports adoption, and paid_campaign_x has much lower advanced_reports adoption than organic. To strengthen the root-cause claim, I should quantify this relationship: churn by advanced_reports adoption overall and within channel/plan, and possibly a simple multivariate control (e.g., by channel + plan group rates). Also inspect whether cancellation reasons for users without advanced_reports skew toward missing_features / switched_competitor. Then I can form a solid key finding and recommendations. Code:

```python
# Quantify relationship between advanced_reports adoption and churn
fa = feature_adoption.copy()
fa["used_advanced_reports"] = fa["advanced_reports"] > 0

print("Overall churn by advanced_reports adoption:")
print(fa.groupby("used_advanced_reports")["is_churned"].agg(["mean", "count"]))

print("\nChurn by channel and advanced_reports adoption:")
channel_adv = fa.groupby(["acquisition_channel", "used_advanced_reports"])["is_churned"].agg(["mean", "count"]).reset_index()
print(channel_adv)

print("\nChurn by plan and advanced_reports adoption:")
plan_adv = fa.groupby(["plan", "used_advanced_reports"])["is_churned"].agg(["mean", "count"]).reset_index()
print(plan_adv)

print("\nPaid Campaign X churn by plan and advanced_reports adoption:")
pcx_plan_adv = (
    fa[fa["acquisition_channel"] == "paid_campaign_x"]
    .groupby(["plan", "used_advanced_reports"])["is_churned"]
    .agg(["mean", "count"])
    .reset_index()
)
print(pcx_plan_adv)

print("\nAdvanced_reports adoption by channel:")
print(fa.groupby("acquisition_channel")["used_advanced_reports"].mean().sort_values())

# Cancellation reasons by advanced_reports usage among churned users
churned_fa = fa[fa["is_churned"]].copy()
reason_by_adv = pd.crosstab(churned_fa["used_advanced_reports"], churned_fa["cancellation_reason"], normalize="index")
print("\nCancellation reasons among churned users by advanced_reports usage:")
print(reason_by_adv)

print("\nPaid Campaign X cancellation reasons by advanced_reports usage:")
pcx_reason_by_adv = pd.crosstab(
    churned_fa[churned_fa["acquisition_channel"]=="paid_campaign_x"]["used_advanced_reports"],
    churned_fa[churned_fa["acquisition_channel"]=="paid_campaign_x"]["cancellation_reason"],
    normalize="index"
)
print(pcx_reason_by_adv)
```

2026/03/08 15:53:03 INFO dspy.predict.rlm: Output (2,852 chars): Overall churn by advanced_reports adoption: mean count used_advanced_reports\
False 0.277490 6948 True 0.147117 3052

Churn by channel and advanced_reports adoption: acquisition_channel used_advanced_reports mean count 0 organic False 0.157841 1964 1 organic True 0.157598 1066 2 paid_campaign_x False 0.550573 2007 3 paid_campaign_x True 0.000000 462 4 paid_campaign_y False 0.138693 995 5 paid_campaign_y True 0.160714 504 6 referral False 0.152664 976 7 referral True 0.150289 519 8 social False 0.224652 1006 9 social True 0.243513 501

Churn by plan and advanced_reports adoption: plan used_advanced_reports mean count 0 enterprise False 0.285118 719 1 enterprise True 0.136054 294 2 free False 0.283103 2681 3 free True 0.144223 1255 4 pro False 0.268606 1478 5 pro True 0.142617 596 6 starter False 0.273913 2070 7 starter True 0.157663 907

Paid Campaign X churn by plan and advanced_reports adoption: plan used_advanced_reports mean count 0 enterprise False 0.541463 205 1 enterprise True 0.000000 43 2 free False 0.561783 785 3 free True 0.000000 182 4 pro False 0.529833 419 5 pro True 0.000000 89 6 starter False 0.553512 598 7 starter True 0.000000 148

Advanced_reports adoption by channel: acquisition_channel paid_campaign_x 0.187120 social 0.332449 paid_campaign_y 0.336224 referral 0.347157 organic 0.351815 Name: used_advanced_reports, dtype: float64

Cancellation reasons among churned users by advanced_reports usage: cancellation_reason missing_features ... too_expensive used_advanced_reports ...\
False 0.159751 ... 0.154564 True 0.131403 ... 0.187082

[2 rows x 6 columns]

Paid Campaign X cancellation reasons by advanced_reports usage: cancellation_reason missing_features ... too_expensive used_advanced_reports ...\
False 0.149321 ... 0.142986

[1 rows x 6 columns]

2026/03/08 15:53:15 INFO dspy.predict.rlm: RLM iteration 5/15 Reasoning: We have a very strong signal, but the perfect 0% churn among paid_campaign_x users who used advanced_reports could be suspicious or still valid. Before submitting, I should verify counts/consistency and formulate a concise key finding. I also want to see if paid_campaign_x differs in advanced_reports adoption over cohorts and whether no users who churned ever used advanced_reports in that channel. Then I can safely submit with overall churn rate, worst channel, root cause, and recommendations focused on acquisition quality/onboarding to advanced_reports. Code:

```python
# Final verification and compact summary stats for submission
print("Overall churn rate check:", base["is_churned"].mean())

worst_channel = channel_stats["churn_rate"].idxmax()
print("Worst channel:", worst_channel)
print(channel_stats.loc[worst_channel])

# Verify the striking advanced_reports pattern
pcx = fa[fa["acquisition_channel"] == "paid_campaign_x"].copy()
print("\nPaid Campaign X advanced_reports usage vs churn crosstab:")
print(pd.crosstab(pcx["used_advanced_reports"], pcx["status"], margins=True))

print("\nPaid Campaign X advanced_reports adoption by signup cohort:")
print(
    pcx.groupby("signup_month")["used_advanced_reports"].mean()
)

print("\nPaid Campaign X churn by cohort and advanced_reports usage:")
print(
    pcx.groupby(["signup_month", "used_advanced_reports"])["is_churned"]
    .agg(["mean", "count"])
)

# Compare campaign_x to others on adoption
adv_by_channel = fa.groupby("acquisition_channel")["used_advanced_reports"].mean().sort_values()
print("\nAdvanced reports adoption by channel:")
print(adv_by_channel)

# Prepare outputs
overall_churn_rate_out = float(base["is_churned"].mean())
worst_channel_out = str(worst_channel)

key_finding_out = (
    "Churn is 23.77% overall, and paid_campaign_x is the clear outlier with 44.76% churn "
    "across every signup cohort and plan. The strongest behavioral signal is early adoption "
    "of advanced_reports: users who adopt it churn far less overall (14.7% vs 27.7%), while "
    "paid_campaign_x has by far the lowest advanced_reports adoption (18.7% vs ~33-35% for other channels). "
    "Within paid_campaign_x, churn is concentrated entirely among users who never adopt advanced_reports, "
    "indicating the drop in retention is primarily an acquisition-quality/onboarding problem rather than "
    "a broad product usage decline."
)

recommendations_out = (
    "1) Audit or pause paid_campaign_x because it is bringing in low-retention users. "
    "2) Redesign onboarding for campaign_x traffic to drive advanced_reports activation in the first 14 days "
    "(guided setup, templates, onboarding emails, success prompts). "
    "3) Make ad messaging and landing pages align with the advanced reporting value proposition so acquired users "
    "match the product's core use case. "
    "4) Track activation and retention by channel weekly, using advanced_reports adoption as the primary leading indicator."
)

print("\nPrepared outputs:")
print(overall_churn_rate_out)
print(worst_channel_out)
print(key_finding_out)
print(recommendations_out)
```

2026/03/08 15:53:24 INFO dspy.predict.rlm: RLM iteration 6/15 Reasoning: The error happened because `fa`/`feature_adoption` was built from `base_cols`, which did not include `signup_month`. I already have enough evidence for the conclusion, but I should fix the verification step cleanly by adding `signup_month` back from `base` and reprinting the final sanity checks. Then, if the outputs look consistent, submit using variables computed in code to avoid retyping. Code:

```python
# Fix missing signup_month for verification
fa = feature_adoption.merge(base[["user_id", "signup_month"]], on="user_id", how="left")

print("Overall churn rate check:", float(base["is_churned"].mean()))

worst_channel = channel_stats["churn_rate"].idxmax()
print("\nWorst channel:", worst_channel)
print(channel_stats.loc[worst_channel])

pcx = fa[fa["acquisition_channel"] == "paid_campaign_x"].copy()

print("\nPaid Campaign X advanced_reports usage vs churn crosstab:")
print(pd.crosstab(pcx["used_advanced_reports"], pcx["status"], margins=True))

print("\nPaid Campaign X advanced_reports adoption by signup cohort:")
print(pcx.groupby("signup_month")["used_advanced_reports"].mean())

print("\nPaid Campaign X churn by cohort and advanced_reports usage:")
print(
    pcx.groupby(["signup_month", "used_advanced_reports"])["is_churned"]
    .agg(["mean", "count"])
)

overall_churn_rate_out = float(base["is_churned"].mean())
worst_channel_out = str(worst_channel)

key_finding_out = (
    "Churn is 23.77% overall, and paid_campaign_x is the clear outlier with 44.76% churn "
    "across every signup cohort and plan. The strongest behavioral signal is early adoption "
    "of advanced_reports: users who adopt it churn far less overall (14.7% vs 27.7%), while "
    "paid_campaign_x has by far the lowest advanced_reports adoption (18.7% vs ~33-35% for other channels). "
    "Within paid_campaign_x, churn is concentrated among users who never adopt advanced_reports, "
    "indicating the retention drop is primarily an acquisition-quality/onboarding problem rather than "
    "a broad product usage decline."
)

recommendations_out = (
    "Pause or tighten paid_campaign_x targeting; redesign onboarding for campaign_x users to drive "
    "advanced_reports activation in the first 14 days; align ad/landing-page messaging to the product's "
    "core reporting use case; and track weekly activation-to-retention funnels by channel with "
    "advanced_reports adoption as the leading indicator."
)

print("\nPrepared outputs:")
print("overall_churn_rate_out =", overall_churn_rate_out)
print("worst_channel_out =", worst_channel_out)
print("key_finding_out =", key_finding_out)
print("recommendations_out =", recommendations_out)

SUBMIT(overall_churn_rate_out, worst_channel_out, key_finding_out, recommendations_out)
```

2026/03/08 15:53:33 INFO dspy.predict.rlm: RLM iteration 7/15 Reasoning: Need fix because fa overwritten without used_advanced_reports. Recreate from feature_adoption/base including signup_month and used_advanced_reports, print sanity checks, then submit. Code:

```python
# Rebuild final verification table cleanly
fa = feature_adoption.merge(
    base[["user_id", "signup_month", "is_churned", "status", "acquisition_channel", "plan"]],
    on=["user_id", "is_churned", "status", "acquisition_channel", "plan"],
    how="left"
).copy()

fa["used_advanced_reports"] = fa["advanced_reports"] > 0

print("Overall churn rate check:", float(base["is_churned"].mean()))

worst_channel = channel_stats["churn_rate"].idxmax()
print("\nWorst channel:", worst_channel)
print(channel_stats.loc[worst_channel])

pcx = fa[fa["acquisition_channel"] == "paid_campaign_x"].copy()

print("\nPaid Campaign X advanced_reports usage vs churn crosstab:")
print(pd.crosstab(pcx["used_advanced_reports"], pcx["status"], margins=True))

print("\nPaid Campaign X advanced_reports adoption by signup cohort:")
print(pcx.groupby("signup_month")["used_advanced_reports"].mean())

print("\nPaid Campaign X churn by cohort and advanced_reports usage:")
print(
    pcx.groupby(["signup_month", "used_advanced_reports"])["is_churned"]
    .agg(["mean", "count"])
)

overall_churn_rate_out = float(base["is_churned"].mean())
worst_channel_out = str(worst_channel)

key_finding_out = (
    "Churn is 23.77% overall, and paid_campaign_x is the clear outlier with 44.76% churn "
    "across every signup cohort and plan. The strongest behavioral signal is early adoption "
    "of advanced_reports: users who adopt it churn far less overall (14.7% vs 27.7%), while "
    "paid_campaign_x has by far the lowest advanced_reports adoption (18.7% vs ~33-35% for other channels). "
    "Within paid_campaign_x, churn is concentrated among users who never adopt advanced_reports, "
    "indicating the retention drop is primarily an acquisition-quality/onboarding problem rather than "
    "a broad product usage decline."
)

recommendations_out = (
    "Pause or tighten paid_campaign_x targeting; redesign onboarding for campaign_x users to drive "
    "advanced_reports activation in the first 14 days; align ad and landing-page messaging to the product's "
    "core reporting use case; and track weekly activation-to-retention funnels by channel with "
    "advanced_reports adoption as the leading indicator."
)

print("\nPrepared outputs:")
print("overall_churn_rate_out =", overall_churn_rate_out)
print("worst_channel_out =", worst_channel_out)
print("key_finding_out =", key_finding_out)
print("recommendations_out =", recommendations_out)

SUBMIT(overall_churn_rate_out, worst_channel_out, key_finding_out, recommendations_out)
```

```

## 7. Results[¶](#7-results)

In [10]:

Copied!

```

print(f"Overall Churn Rate: {result.overall_churn_rate:.1%}") print(f"Worst Channel: {result.worst_channel}") print(f"\\nKey Finding:\\n {result.key_finding}") print(f"\\nRecommendations:\\n {result.recommendations}") print(f"\\nRLM completed in {len(result.trajectory)} iterations")

```

print(f"Overall Churn Rate: {result.overall_churn_rate:.1%}")
print(f"Worst Channel: {result.worst_channel}")
print(f"\nKey Finding:\n {result.key_finding}")
print(f"\nRecommendations:\n {result.recommendations}")
print(f"\nRLM completed in {len(result.trajectory)} iterations")

```

Overall Churn Rate: 23.8% Worst Channel: paid_campaign_x

Key Finding: Churn is 23.77% overall, and paid_campaign_x is the clear outlier with 44.76% churn across every signup cohort and plan. The strongest behavioral signal is early adoption of advanced_reports: users who adopt it churn far less overall (14.7% vs 27.7%), while paid_campaign_x has by far the lowest advanced_reports adoption (18.7% vs ~33-35% for other channels). Within paid_campaign_x, churn is concentrated among users who never adopt advanced_reports, indicating the retention drop is primarily an acquisition-quality/onboarding problem rather than a broad product usage decline.

Recommendations: Pause or tighten paid_campaign_x targeting; redesign onboarding for campaign_x users to drive advanced_reports activation in the first 14 days; align ad and landing-page messaging to the product's core reporting use case; and track weekly activation-to-retention funnels by channel with advanced_reports adoption as the leading indicator.

RLM completed in 7 iterations

```
```
