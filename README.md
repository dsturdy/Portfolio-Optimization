# Conceptual Proof – Asset Allocation Optimization Program

This repository contains my solution for the First Eagle take-home case study:
a rules-based asset allocation optimizer paired with a minimal Streamlit UI
for running scenarios and visualizing results.

The objective of the project is to demonstrate a transparent, repeatable portfolio
construction process built on explicit rules and constraints rather than
discretionary adjustments.

---

## High-level pipeline

At a high level, the workflow is:

1. **Read inputs** from the provided `case_inputs.xlsx`
   - current portfolio weights
   - investable universe
   - asset-group classifications and constraint bands

2. **Build a spliced return history** for funds with short lived histories
   - ETF proxies and proxy blends are defined ahead of time (ex-ante) based on economic exposure,
     not selected after observing performance
   - the best proxy is selected using historical correlation and data coverage
   - pre-inception history is backfilled using a first-overlap price ratio

3. **Estimate return and risk moments**
   - monthly returns derived from the spliced price history
   - expected returns and covariances inferred from historical data using a consistent window

4. **Run a constrained mean–variance optimizer**
   - formulated as a convex optimization problem
   - solved using `cvxpy` (Convex Optimization in Python)
   - all constraints are explicit, rules-based, and reproducible

5. **Visualize outputs in a small UI**
   - current vs optimized portfolio allocations
   - risk contribution by asset group
   - historical rebalanced backtests

---

## Repository layout

- `case_inputs.xlsx`  
  Case file provided by First Eagle.

- `build_spliced_dataset.py`  
  Constructs the historical return dataset used by the optimizer:
  - downloads adjusted price data via `yfinance`
  - constructs ETF and blended proxies
  - selects the best proxy using correlation and proxy data coverage
  - splices pre-inception history using a **first overlapping price ratio**
    (the proxy is scaled to exactly match the fund on the first shared date,
    then used to backfill earlier history)

- `case_io.py`  
  Handles all parsing of the Excel case file:
  - current portfolio weights
  - asset-group labels and min/max allocation bands
  - investable universe definition

- `ticker_meta.py`  
  Stores fund-level metadata (asset class, liquidity, etc.), used to express
  portfolio constraints in a readable, auditable way.

- `optimize_portfolio.py`  
  Core optimization and backtesting logic:
  - estimates return and risk inputs from monthly returns
  - formulates the constrained mean–variance objective
  - solves the optimization using `cvxpy`
  - runs quarterly rebalancing backtests using daily NAV series
  

  **Note:** Key policy constraints (e.g., risk targets, equity caps,
  illiquid caps, cash caps) are intentionally exposed as editable constants
  near the top of this file. When running the app locally, changing these values
  and rerunning the optimization in the UI will immediately reflect the updated constraints.

- `analytics.py`  
  Shared analytics utilities:
  - current portfolio statistics
  - asset-group risk contributions
  - max drawdown calculations


- `app.py`  
  Streamlit UI (Part 2 of the case):
  - sidebar controls for risk profile, objectives, and constraints
  - tables and charts comparing current vs optimized portfolios
  

The UI also includes configurable defaults (such as the lookback window and rebalancing frequency) 
that are defined directly in the code and can be adjusted by editing app.py and rerunning the 
optimization in the UI. 

- `run_case_study.py`  
  One-command driver script:
  - runs `build_spliced_dataset.py`
  - runs the optimizer
  - writes all generated datasets and optimization outputs to `./data/`

---

## Generated outputs

On first run, the `data/` folder is updated containing:

- `chosen_proxies_for_splicing.csv`  
  Documentation of which proxy was selected for each spliced fund,
  including correlations, overlap length, and scaling factors.

- `spliced_price_panel_base_funds.csv`  
  Daily price history (spliced where necessary) for all base funds.

- `spliced_monthly_returns.csv`  
  Monthly returns used by the optimizer.

- `optimized_weights.csv`  
  Final optimized allocations for the selected scenario.

---
## Setup

> **First:** download the project from Github and make sure you are in the project folder
> (the one containing `run_case_study.py` and `app.py`). If you downloaded a ZIP, unzip it and
> open a terminal in that folder (or `cd` into the pathname):
>
> ```bash
> cd First_Eagle_Case_Study
> ```

Create a virtual environment and install dependencies:

### macOS / Linux
1) Create a local, isolated Python environment for the project.  
2) Activate that environment so your shell uses it.  
3) Install all required packages into the environment.
```bash
python -m venv fe_case_env   # use python3 here if python is not found
source fe_case_env/bin/activate
pip install -r requirements.txt
```

### Windows

```bat
python -m venv fe_case_env
fe_case_env\Scripts\activate
pip install -r requirements.txt
```

---

## Run the project

### macOS & Windows

#### Build the dataset & run the optimizer

```bash
python run_case_study.py
```

#### Launch the UI

```bash
python -m streamlit run app.py
```

