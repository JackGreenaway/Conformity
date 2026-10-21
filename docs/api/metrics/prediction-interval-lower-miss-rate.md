# prediction_interval_lower_miss_rate {#prediction-interval-lower-miss-rate}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L320">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_lower_miss_rate</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of targets below their closed prediction intervals.

## What it tells you

This is how often the true target falls below the interval. It reveals misses on the low side; for equal-tail asymmetric intervals, compare it with alpha / 2 as a noisy diagnostic.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One finite numeric target per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for evaluation only.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of targets strictly below the lower bound.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_interval_lower_miss_rate
>>> prediction_interval_lower_miss_rate([0, 2], [[1, 3], [1, 3]])
0.5
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values or weights are invalid.</p></dd>
</dl>
