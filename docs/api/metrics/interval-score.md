# interval_score {#interval-score}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L489">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">interval_score</span>(
    y_true,
    prediction_intervals,
    alpha=0.05,
    *,
    sample_weight=None,
)</code></pre></div>

Mean Winkler score: width plus 2/alpha times each missed-bound distance.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">default=0.05</span></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean interval score; may be infinite. Smaller values indicate a better score.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import interval_score
>>> interval_score([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]], alpha=0.1)
12.0
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
