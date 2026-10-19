# Runnable examples

Choose a question before choosing a component. Every Python block in the new guides is standalone and executed during validation; examples use synthetic data or local arrays and need no external downloads.

| Question | Example | What to inspect |
| --- | --- | --- |
| How wide is uncertainty for a numeric prediction? | [Getting started](../getting-started.md) | Test coverage and mean width |
| Does an interval cross a decision boundary? | [Regression decisions](regression.md) | Review count and interval score |
| Which labels remain plausible? | [Classification review](classification.md) | Class mapping, coverage and set size |
| What do intervals and sets look like? | [Visual examples](plots.md) | Held-out intervals and prediction set sizes |
| How should I compare coverage and efficiency? | [Diagnostics](../user-guide/diagnostics.md) | Interval tail misses, interval scores, set sizes and class-wise coverage |
| Can I develop models with sklearn tools? | [Cross-validation](cross-validation.md) | Clone lifecycle and point scores |
| Where are outcomes missed? | [Plotting](../plotting.md) | Predicted ranges and observed test targets |

The [original workflows](../legacy/README.md) remain available, including supported outer-pipeline examples and their statistical caveats. Run examples with the locked environment described in [development](../development.md); keep calibration independent of all learning.
