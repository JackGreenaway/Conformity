# API reference

Public imports are available from `conformity`. Signatures, parameter defaults,
fitted attributes and method contracts below are generated from the implementation.

For interpretation and evaluation choices, read [diagnostics](user-guide/diagnostics.md).
Set metrics use boolean membership in explicit fitted class order; interval metrics
use closed lower/upper bounds. Calibration is unweighted even when evaluation uses
weights. See [calibration assumptions](user-guide/calibration.md) and the
[examples gallery](examples/index.md) for complete workflows.

## Classes

| Class | Reference |
| --- | --- |
| `BaseConformalPredictor` | [BaseConformalPredictor](classes/base-conformal-predictor.md) |
| `ConformalRegressor` | [ConformalRegressor](classes/conformal-regressor.md) |
| `ConformalClassifier` | [ConformalClassifier](classes/conformal-classifier.md) |

## Functions

<!-- BEGIN GENERATED REFERENCE -->
## prediction_set_coverage {#prediction-set-coverage}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L74">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_coverage</span>(
    y_true,
    prediction_set,
    *,
    classes=None,
    sample_weight=None,
)</code></pre></div>

Fraction of sets containing the true class; supports arbitrary labels.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">classes</code> <span class="api-field-type">default=None</span></dt><dd><p>Unique class labels in boolean membership-column order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of samples whose set contains the true label; weighted when requested.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_set_coverage
>>> sets = [[True, False], [False, True], [True, True]]
>>> prediction_set_coverage(["billing", "account", "billing"], sets,
...                         classes=["billing", "account"])
1.0
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## prediction_set_coverage_by_class {#prediction-set-coverage-by-class}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L123">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_coverage_by_class</span>(
    y_true,
    prediction_set,
    *,
    classes,
    sample_weight=None,
)</code></pre></div>

Return empirical set coverage for each represented class.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in <code>classes</code> order.</p></dd>
<dt><code class="api-field-name">classes</code></dt><dd><p>Unique class labels in boolean membership-column order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for evaluation only.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">dict</span></dt><dd><p>Empirical coverage keyed by class label.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_set_coverage_by_class
>>> sets = [[True, False], [False, True], [True, False]]
>>> prediction_set_coverage_by_class(
...     ["a", "b", "b"], sets, classes=["a", "b"]
... )
{'a': 1.0, 'b': 0.5}
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If shapes, labels, weights or class representation are invalid.</p></dd>
</dl>

## prediction_interval_coverage {#prediction-interval-coverage}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L282">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_coverage</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of targets contained in closed prediction intervals.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of targets inside their closed interval; weighted when requested.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_interval_coverage
>>> prediction_interval_coverage([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]])
0.5
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## prediction_interval_lower_miss_rate {#prediction-interval-lower-miss-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L320">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_lower_miss_rate</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of targets below their closed prediction intervals.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One finite numeric target per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for evaluation only.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of targets strictly below the lower bound.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_interval_lower_miss_rate
>>> prediction_interval_lower_miss_rate([0, 2], [[1, 3], [1, 3]])
0.5
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values or weights are invalid.</p></dd>
</dl>

## prediction_interval_upper_miss_rate {#prediction-interval-upper-miss-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L364">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_upper_miss_rate</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of targets above their closed prediction intervals.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One finite numeric target per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for evaluation only.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of targets strictly above the upper bound.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_interval_upper_miss_rate
>>> prediction_interval_upper_miss_rate([0, 4], [[1, 3], [1, 3]])
0.5
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values or weights are invalid.</p></dd>
</dl>

## prediction_interval_width {#prediction-interval-width}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L408">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_interval_width</span>(
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Mean interval width; unbounded intervals have infinite width.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean upper-minus-lower bound width; may be infinite.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_interval_width
>>> prediction_interval_width([[0.0, 2.0], [1.0, 3.0]])
2.0
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## normalized_prediction_interval_width {#normalized-prediction-interval-width}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L440">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">normalized_prediction_interval_width</span>(
    y_true,
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Mean interval width divided by the target's interquartile range.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One finite numeric target per sample; its unweighted IQR is the scale.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights for averaging interval widths.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean width divided by the unweighted target IQR; may be infinite.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import normalized_prediction_interval_width
>>> normalized_prediction_interval_width(
...     [0, 0, 2, 2], [[0, 2], [0, 2], [0, 2], [0, 2]]
... )
1.0
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values or weights are invalid.</p></dd>
</dl>

## interval_score {#interval-score}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L489">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">interval_score</span>(
    y_true,
    prediction_intervals,
    alpha=0.05,
    *,
    sample_weight=None,
)</code></pre></div>

Mean Winkler score: width plus 2/alpha times each missed-bound distance.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">default=0.05</span></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean interval score; may be infinite. Smaller values indicate a better score.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import interval_score
>>> interval_score([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]], alpha=0.1)
12.0
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## prediction_set_size {#prediction-set-size}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L189">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_size</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Mean cardinality (smaller is more efficient at comparable coverage).

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Mean number of included labels per sample; weighted when requested.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_set_size
>>> prediction_set_size([[True, False], [False, False], [True, True]])
1.0
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## prediction_set_empty_rate {#prediction-set-empty-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L220">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_empty_rate</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of empty sets.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of samples with no included labels; weighted when requested.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_set_empty_rate
>>> prediction_set_empty_rate([[True, False], [False, False]])
0.5
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

## prediction_set_singleton_rate {#prediction-set-singleton-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L251">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module">conformity.</span><span class="api-name">prediction_set_singleton_rate</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of sets containing exactly one label.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

### Returns

<dl class="api-fields">
<dt><code class="api-field-name">result</code> <span class="api-field-type">float</span></dt><dd><p>Fraction of samples with exactly one included label; weighted when requested.</p></dd>
</dl>

### Examples

```pycon
>>> from conformity import prediction_set_singleton_rate
>>> prediction_set_singleton_rate([[True, False], [True, True]])
0.5
```

### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

<!-- END GENERATED REFERENCE -->
