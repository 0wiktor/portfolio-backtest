"""pbt — a small, reproducible equity portfolio backtester.

The package is deliberately thin: one module per stage of the pipeline, and no
shared mutable state. A run flows config -> data -> weights -> engine ->
metrics/montecarlo -> plots/report.
"""

__version__ = "0.1.0"
