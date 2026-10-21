# normalized_prediction_interval_width {#normalized-prediction-interval-width}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L440">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">normalized_prediction_interval_width</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Mean interval width divided by the target's interquartile range.

## What it tells you

This expresses average interval width relative to the test targets' interquartile range, making widths easier to compare across target scales. It measures size, not coverage, and is infinite when the IQR is zero.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One finite numeric target per sample; its unweighted IQR is the scale.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for averaging interval widths.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean width divided by the unweighted target IQR; may be infinite.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import normalized_prediction_interval_width
>>> normalized_prediction_interval_width(
...     [0, 0, 2, 2], [[0, 2], [0, 2], [0, 2], [0, 2]]
... )
1.0
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values or weights are invalid.</p></dd>
</dl>
