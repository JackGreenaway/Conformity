# prediction_set_empty_rate {#prediction-set-empty-rate}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L220">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_empty_rate</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of empty sets.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of samples with no included labels; weighted when requested.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_set_empty_rate
>>> prediction_set_empty_rate([[True, False], [False, False]])
0.5
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
