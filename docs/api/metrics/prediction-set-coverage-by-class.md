# prediction_set_coverage_by_class {#prediction-set-coverage-by-class}

[Metrics overview](../metrics.md)

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L123">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_coverage_by_class</span>(
    y_true,
    prediction_set,
    *,
    classes,
    sample_weight=None,
)</code></pre></div>

Return empirical set coverage for each represented class.

## What it tells you

For each true class, this shows how often its examples were retained in their prediction sets. It helps reveal uneven empirical performance, but does not create a class-conditional coverage guarantee.

## Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in <code>classes</code> order.</p></dd>
<dt><code class="api-field-name">classes</code></dt><dd><p>Unique class labels in boolean membership-column order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for evaluation only.</p></dd>
</dl>

## Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">dict</span></dt><dd><p>Empirical coverage keyed by class label.</p></dd>
</dl>

## Examples

```pycon
>>> from conformity import prediction_set_coverage_by_class
>>> sets = [[True, False], [False, True], [True, False]]
>>> prediction_set_coverage_by_class(
...     ["a", "b", "b"], sets, classes=["a", "b"]
... )
{'a': 1.0, 'b': 0.5}
```

## Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If shapes, labels, weights or class representation are invalid.</p></dd>
</dl>
