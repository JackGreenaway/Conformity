# prediction_interval_coverage {#prediction-interval-coverage}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L282">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_coverage</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of targets contained in closed prediction intervals.

## What it tells you

This is the share of evaluated targets inside their intervals. It is an empirical test-set summary; a finite test sample can fall below or above the nominal coverage target.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of targets inside their closed interval; weighted when requested.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_interval_coverage
>>> prediction_interval_coverage([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]])
0.5
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
