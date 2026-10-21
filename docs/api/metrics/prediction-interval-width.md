# prediction_interval_width {#prediction-interval-width}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L408">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_width</span>(
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Mean interval width; unbounded intervals have infinite width.

## What it tells you

This is the average distance from the lower to upper bound, in the target's original units. Smaller intervals are more precise only when coverage remains adequate; an infinite result means some intervals are unbounded.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean upper-minus-lower bound width; may be infinite.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_interval_width
>>> prediction_interval_width([[0.0, 2.0], [1.0, 3.0]])
2.0
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
