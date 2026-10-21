# prediction_set_size {#prediction-set-size}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L189">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_size</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Mean cardinality (smaller is more efficient at comparable coverage).

## What it tells you

This is the average number of labels offered per example. Smaller sets are more decisive, but compare them alongside coverage so a set is not judged useful just because it leaves labels out.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean number of included labels per sample; weighted when requested.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_set_size
>>> prediction_set_size([[True, False], [False, False], [True, True]])
1.0
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
