# portfolio-backtest

A backtesting tool for equity portfolios. You give it a list of tickers, a weighting
rule and a rebalancing schedule. It simulates holding that portfolio, charges
transaction costs, and compares the result against a benchmark.

I built this because I wanted something concrete behind the portfolio construction
ideas I write about in applications. It is small on purpose: three weighting schemes,
fourteen metrics, five charts. My first attempt was a 2,000 line notebook with thirty
metrics and fourteen charts, and it was worse. Most of those metrics were variations
on each other, and having thirty of them mostly gave me room to quote whichever one
happened to look good.

## Install and run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

That is the Windows path. On macOS or Linux it is `.venv/bin/python`.

```bash
.venv/Scripts/pbt run --config config/example.yaml
```

This prints a report to the terminal and writes it to `output/report.txt`, along with
five charts. Prices are cached to `data_cache/` as parquet files, so the first run
downloads from Yahoo and every run after that with the same tickers and dates works
offline.

To run the tests:

```bash
.venv/Scripts/python -m pytest
```

Python 3.11 or newer. It uses pandas, numpy, yfinance, matplotlib, pyyaml, scipy and
pyarrow, plus pytest for the tests.

### Command line options

| Flag | Default | What it does |
|---|---|---|
| `--config` | required | path to the YAML config |
| `--output-dir` | `output` | where charts and the report go |
| `--cache-dir` | `data_cache` | the price cache |
| `--seed` | `0` | Monte Carlo seed, fixed so runs reproduce |
| `--paths` | `10000` | number of simulated paths |

These are flags rather than config keys because they describe how you are running the
tool, not what the backtest is. The config file is the specification of the backtest
itself, and it rejects any key it does not recognise so that list stays meaningful.

## Configuration

Everything about a run lives in the YAML file, so a result can be reproduced from that
file alone. See `config/example.yaml`.

| Key | Units or values | Required |
|---|---|---|
| `tickers` | list of symbols | yes |
| `benchmark` | one symbol | no, defaults to `SPY` |
| `start`, `end` | `YYYY-MM-DD`, both inclusive | yes |
| `initial_capital` | currency units | yes |
| `weighting` | `equal`, `custom` or `inverse_vol` | yes |
| `custom_weights` | list summing to 1.0, same order as `tickers` | only for `custom` |
| `lookback_days` | trading days | only for `inverse_vol` |
| `rebalance` | `none`, `monthly`, `quarterly` or `annual` | yes |
| `risk_free_rate` | annual decimal, so `0.02` is 2% a year | yes |
| `transaction_cost_bps` | basis points charged on turnover | yes |

The config is checked when it loads. Unknown keys, missing keys, weights that do not
sum to one, weights that do not match the number of tickers, and dates in the wrong
order each produce a single line explaining the problem.

## How it works

**Prices.** I use yfinance with `auto_adjust=True`, which folds dividends and splits
into the Close column. There is no reference anywhere to `Adj Close`. With
`auto_adjust` switched on, Close is already the adjusted series, and using both would
count the adjustment twice.

**Missing data.** Gaps of up to five trading days get forward filled. If a ticker is
still missing more than 10% of the window after that, it is dropped and the reason is
printed. If a ticker's history starts after the configured `start` date, the run stops
with an error rather than quietly shortening the window, because a shortened window
measures something different from what you asked for.

**Weighting.** Equal is 1/N. Custom takes your weights and normalises them. Inverse
volatility weights each holding by 1 divided by its standard deviation of daily
returns over the trailing `lookback_days` ending on the rebalance date. One of the
tests changes every observation after the rebalance date and checks the weights do not
move.

**There is no market cap weighting, and that is on purpose.** yfinance only gives you
current shares outstanding. Building historical market caps from that means applying
today's share counts to past prices, which is lookahead bias sitting directly in the
weights. It would make the results look better than they should.

**Warm-up.** Inverse volatility needs history before it can produce a first estimate,
and I take that history from inside the configured window rather than quietly
downloading extra data from before the start date. So the first `lookback_days` are
spent warming up and the equity curve begins after them. The benchmark is cut to the
same start date so the comparison stays fair.

**Rebalancing** happens at the close of the last actual trading day of each period.
The dates come from grouping the observed trading calendar, not from generating
calendar month ends, plenty of which land on weekends. The first and last days of the
sample are never rebalance dates: entry is handled separately, and a trade on the
final day cannot affect any return after it, so charging for it would only make the
result look worse for no reason.

**Transaction costs.** Turnover at a rebalance is half the sum of the absolute
differences between target and drifted weights. The half is there because every sale
funds a purchase, so adding both sides counts each trade twice. The cost is turnover
times the basis point rate times portfolio value, charged on the day it happens, which
means the daily returns are already net of it.

I charge the initial purchase at turnover 1.0 rather than 0.5. The half formula
assumes you are already fully invested on both sides, but at entry the starting
weights are all zero, so using it would charge half the cost of actually buying the
portfolio.

**Annualisation** uses 252 trading days throughout, defined in one place. The risk
free rate is converted from annual to daily by compounding rather than by dividing
by 252.

**Sortino** measures downside deviation against the daily risk free rate and divides
by the full sample size, not by the number of losing days. Dividing by the smaller
count is a common variant that produces a higher number, and the two are not
comparable, so it is worth knowing which one you are looking at.

**Beta, alpha and R-squared** all come out of one regression of daily portfolio excess
returns on daily benchmark excess returns, so they are always consistent with each
other. Alpha is annualised by multiplying the daily intercept by 252.

**VaR and CVaR** are historical rather than parametric: the empirical 5th percentile
of daily returns, and the average of the returns beyond it. Both are reported as
positive numbers, so 1.8% means a loss of 1.8%. Maximum drawdown is reported as a
negative number.

**Monte Carlo** uses a stationary block bootstrap. It resamples blocks of consecutive
historical days with geometric lengths averaging 21 days, which keeps volatility
clustering intact. Resampling single days at random would throw that away.

The portfolio and the benchmark are resampled together, using one set of dates applied
to both. Every simulated day pairs the portfolio's actual return with the benchmark's
return from that same historical day. If you simulated them separately you would break
their correlation, and "probability the portfolio beats the benchmark" would just be
comparing two unrelated random walks.

**The report does not grade anything.** No letter, no stars, no Excellent or Poor. It
reports the numbers and states whether the portfolio beat the benchmark on total
return and on Sharpe. A backtest of a ticker list somebody picked by hand is not
enough evidence to grade a strategy, and putting a confident label on it would suggest
otherwise.

## Known limitations

These matter more than the features. Read them before drawing conclusions from
anything this produces.

**Survivorship and selection bias.** This is the biggest problem and it is not
fixable here. You choose the tickers today, already knowing which companies survived
and which did well. Backtesting a hand-picked list of current large caps will
overstate performance, often by a lot, because the failures and the delisted names
were never in your list to begin with. The engine has no point-in-time index
membership data, so it cannot know what you would have chosen at the start date.
Every number it produces is conditional on a selection you made with hindsight. A
backtest that looks good over a list you picked yourself is close to no evidence at
all.

**No modelling of delistings, corporate actions or dividend timing** beyond whatever
yfinance's adjustment already handles. Dividends are effectively treated as reinvested
instantly and for free, when in reality they arrive on a settlement date, may be
taxed, and have to be reinvested at a real price. Delisted and acquired tickers cannot
be downloaded at all, which is survivorship bias in the data source rather than in my
code.

**The cost model ignores market impact, spread and slippage.** It charges a flat fee
in basis points on turnover. Real trading pays the bid-ask spread, moves the price
when the order is large relative to volume, and does not fill at the closing price the
model assumes. For a small account in liquid ETFs the gap is modest. For size, or for
illiquid names, this understates costs, and it understates them more the more the
strategy trades, so it flatters high turnover strategies specifically.

**The benchmark pays no costs at all.** It is a frictionless buy and hold, while the
portfolio pays to enter and to rebalance. That makes the comparison harder for the
portfolio, which is the safer direction to be wrong in, but it is not equal treatment
and I would not present it as such.

**Trades happen at the closing price** on the rebalance date. There is no intraday
data and no gap modelled between deciding to trade and trading.

**No tax of any kind.** No capital gains, no dividend withholding. A taxable account
would keep meaningfully less than these figures show.

**The Monte Carlo resamples the past.** It assumes the distribution of returns and
their dependence structure carry forward unchanged, which is exactly what a regime
change breaks. The percentile bands describe reshufflings of the history you gave it,
not the range of things that could happen. Ten years containing one long bull market
will produce optimistic simulations however many paths you draw.

**Free data has errors.** yfinance is an unofficial interface to Yahoo Finance.
Adjustments, splits and older history contain occasional mistakes and nobody is
guaranteeing any of it.

**One market, one sample.** This is a single historical path through a single market.
None of the metrics come with confidence intervals, and a Sharpe ratio measured over
ten years has a wide standard error, so a gap of a tenth or two between portfolio and
benchmark is not evidence of skill.

## Layout

```
config/example.yaml    the run configuration
src/pbt/
  config.py            loads and validates the YAML
  data.py              yfinance download, parquet cache, missing data rules
  weights.py           equal, custom and inverse volatility schemes
  engine.py            the backtest loop, rebalancing and costs
  metrics.py           the fourteen metrics
  montecarlo.py        joint stationary block bootstrap
  plots.py             the five charts
  report.py            comparison table, drawdowns, text summary
  cli.py               command line entry point
tests/                 pytest, synthetic data, no network access
notebooks/demo.ipynb   a walkthrough of a full run
```

## Tests

The tests run offline against synthetic data. Metrics are checked against answers
worked out by hand rather than against the library's own output, since a metric that
agrees with itself only proves it is consistent.

They cover the exact CAGR and zero volatility you should get from a constant return
series, turnover and cost arithmetic on a two asset case I worked through on paper,
rebalance dates landing on real trading days when the calendar month end falls on a
weekend, and the inverse volatility lookahead check described above.
