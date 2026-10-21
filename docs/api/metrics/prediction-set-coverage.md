# prediction_set_coverage {#prediction-set-coverage}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L74">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_coverage</span>(
    y_true,
    prediction_set,
    *,
    classes=None,
    sample_weight=None,
)</code></pre></div>

Fraction of sets containing the true class; supports arbitrary labels.

## What it tells you

On the evaluated examples, this is the share whose true label appears in the returned set. It summarizes overall set coverage; it does not promise that this particular finite batch reaches the requested target.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">classes</code> <span class="api-field-type">default=None</span></dt><dd><p>Unique class labels in boolean membership-column order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of samples whose set contains the true label; weighted when requested.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_set_coverage
>>> sets = [[True, False], [False, True], [True, True]]
>>> prediction_set_coverage(["billing", "account", "billing"], sets,
...                         classes=["billing", "account"])
1.0
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>
