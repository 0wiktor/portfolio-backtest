# portfolio-backtest

A backtesting tool for equity portfolios. You give it a list of tickers, a weighting
rule and a rebalancing schedule. It simulates holding that portfolio, charges
transaction costs, and compares the result against a benchmark.

I built it to have something concrete behind the portfolio construction ideas I write
about in applications. It is small on purpose: three weighting schemes, fourteen
metrics, five charts. My first attempt was a 2,000 line notebook with thirty metrics,
and it was worse. Most of them were variations on each other, and having thirty mostly
gave me room to quote whichever one happened to look good.

## Install and run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/pbt run --config config/example.yaml
```

Those are Windows paths; on macOS or Linux it is `.venv/bin/`. Needs Python 3.11 or
newer.

This prints a report, writes it to `output/report.txt`, and saves five charts next to
it. Prices are cached to `data_cache/` as parquet, so the first run downloads from
Yahoo and everything after that works offline. There are `--output-dir`, `--cache-dir`,
`--seed` and `--paths` flags if you want to move things around or change the number of
simulated paths.

Tests are `.venv/Scripts/python -m pytest`. They run offline against synthetic data,
and the metrics are checked against answers worked out by hand rather than against the
library's own output.

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

The config is checked when it loads, so unknown keys, missing keys, weights that do
not sum to one and dates in the wrong order each produce a single line explaining the
problem.

## Methodology

**Prices** come from yfinance with `auto_adjust=True`, which folds dividends and
splits into the Close column. There is no reference anywhere to `Adj Close`, since
using both would count the adjustment twice.

**Missing data.** Gaps of up to five trading days are forward filled. A ticker still
missing more than 10% of the window is dropped, with the reason printed. If a ticker's
history starts after the configured start date the run stops rather than quietly
shortening the window, because a shortened window measures something different from
what you asked for.

**Weighting** is 1/N, your own weights normalised, or inverse volatility. The last one
weights each holding by 1 divided by its standard deviation of daily returns over the
trailing `lookback_days` ending on the rebalance date. One of the tests changes every
observation after that date and checks the weights do not move.

**There is no market cap weighting, on purpose.** yfinance only exposes current shares
outstanding, so building historical market caps from it means applying today's share
counts to past prices. That is lookahead bias sitting directly in the weights, and it
would make results look better than they should.

Inverse volatility needs history before its first estimate, and I take that from inside
the configured window rather than downloading extra data from before the start date. So
the first `lookback_days` are spent warming up and the curve begins after them, with the
benchmark cut to the same start so the comparison stays fair.

**Rebalancing** happens at the close of the last actual trading day of each period,
found by grouping the observed trading calendar rather than generating calendar month
ends, plenty of which fall on weekends. Turnover is half the sum of absolute
differences between target and drifted weights, since every sale funds a purchase and
adding both sides counts each trade twice. The initial purchase is charged at turnover
1.0 rather than 0.5, because the half formula assumes you are already invested on both
sides and at entry the starting weights are all zero.

**Conventions.** Annualisation uses 252 trading days, defined in one place, and the
risk free rate converts from annual to daily by compounding rather than dividing.
Sortino divides by the full sample size, not by the count of losing days, which is a
common variant that produces a higher number. Beta, alpha and R-squared come out of one
regression on daily excess returns so they stay consistent with each other. VaR and
CVaR are historical rather than parametric and are reported as positive numbers, so
1.8% means a loss of 1.8%; maximum drawdown is reported as negative.

**Monte Carlo** uses a stationary block bootstrap, resampling blocks of consecutive
days averaging 21 long so volatility clustering survives. The portfolio and benchmark
are resampled on the same dates. Simulating them separately would break their
correlation, and "probability the portfolio beats the benchmark" would just be
comparing two unrelated random walks.

**The report does not grade anything.** It reports numbers and states whether the
portfolio beat the benchmark on total return and on Sharpe. A backtest of a hand-picked
ticker list is not enough evidence to grade a strategy.

## Known limitations

These matter more than the features.

**Survivorship and selection bias.** The biggest problem here, and not fixable. You
choose the tickers today, already knowing which companies survived and which did well.
Backtesting a hand-picked list of current large caps will overstate performance, often
by a lot, because the failures and the delisted names were never in the list to begin
with. The engine has no point-in-time index membership data, so it cannot know what you
would have chosen at the start date. Every number is conditional on a selection you
made with hindsight, and a backtest that looks good over a list you picked yourself is
close to no evidence at all.

**No modelling of delistings, corporate actions or dividend timing** beyond what
yfinance's adjustment already handles. Dividends are effectively treated as reinvested
instantly and for free, when really they arrive on a settlement date, may be taxed, and
have to be reinvested at a real price. Delisted and acquired tickers cannot be
downloaded at all, which is survivorship bias in the data source rather than in my code.

**The cost model ignores market impact, spread and slippage.** It charges a flat fee in
basis points on turnover. Real trading pays the bid-ask spread, moves the price when the
order is large relative to volume, and does not fill at the closing price the model
assumes. For a small account in liquid ETFs the gap is modest, but for size or illiquid
names this understates costs, and it understates them more the more the strategy trades,
so it flatters high turnover strategies specifically.

**The benchmark pays no costs**, while the portfolio pays to enter and rebalance. That
makes the comparison harder for the portfolio, which is the safer direction, but it is
not equal treatment.

**No tax, and no intraday data.** Trades happen at the closing price with no gap
modelled between deciding and trading, and there is no capital gains or dividend
withholding, so a taxable account would keep meaningfully less than these figures show.

**The Monte Carlo resamples the past.** It assumes the distribution of returns carries
forward unchanged, which is what a regime change breaks. The bands describe reshufflings
of the history you gave it, not the range of things that could happen. Ten years
containing one long bull market produce optimistic simulations however many paths you
draw.

**One market, one sample, and free data.** yfinance is an unofficial interface to Yahoo
Finance and its older history contains occasional errors. None of the metrics come with
confidence intervals, and a Sharpe ratio measured over ten years has a wide standard
error, so a gap of a tenth or two between portfolio and benchmark is not evidence of
skill.
