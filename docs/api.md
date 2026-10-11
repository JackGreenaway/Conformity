# API reference

Import the public API from `conformity`. Signatures, parameter descriptions, fitted attributes, properties and public methods below and on the class pages are generated from the live implementation. Source links point to this repository or to the inherited scikit-learn implementation.

## Choosing a component

| Component | When to use it | Reference |
| --- | --- | --- |
| `BaseConformalPredictor` | Extend or understand the calibration lifecycle | [Class reference](classes/base-conformal-predictor.md) |
| `ConformalRegressor` | Numeric outcome intervals | [Class reference](classes/conformal-regressor.md) |
| `ConformalClassifier` | Categorical prediction sets | [Class reference](classes/conformal-classifier.md) |
| Public metrics | Evaluate coverage and efficiency on independent data | Definitions below |

## Shared estimator behavior

Constructors retain configuration, `fit` trains a fresh clone, and `set_params` invalidates fitted/calibrated state. Point predictions work after fitting; uncertainty output needs calibration. `score` measures point performance. Fit and evaluation weights do not weight calibration. See [pipelines](user-guide/estimators.md) and [calibration limits](user-guide/calibration.md).

<!-- BEGIN GENERATED REFERENCE -->
## prediction_set_coverage {#prediction-set-coverage}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L74">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_set_coverage</span>(
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
<dt><code class="api-field-name">classes</code></dt><dd><p>Unique class labels in boolean membership-column order.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## prediction_interval_coverage {#prediction-interval-coverage}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L150">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_interval_coverage</span>(
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
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## prediction_interval_width {#prediction-interval-width}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L172">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_interval_width</span>(
    prediction_intervals,
    *,
    sample_weight=None,
)</code></pre></div>

Mean interval width; unbounded intervals have infinite width.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## interval_score {#interval-score}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L188">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">interval_score</span>(
    y_true,
    prediction_intervals,
    alpha=0.05,
    *,
    sample_weight=None,
)</code></pre></div>

Mean Winkler score: width plus 2/alpha times each missed-bound distance.

    Lower is better; combines sharpness and miscoverage. Infinite intervals
    have infinite score. Alpha must match that used to construct intervals.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">y_true</code></dt><dd><p>One observed target or class label per sample.</p></dd>
<dt><code class="api-field-name">prediction_intervals</code></dt><dd><p>Ordered lower/upper bounds of shape (n_samples, 2).</p></dd>
<dt><code class="api-field-name">alpha</code></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## prediction_set_size {#prediction-set-size}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L105">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_set_size</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Mean cardinality (smaller is more efficient at comparable coverage).

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## prediction_set_empty_rate {#prediction-set-empty-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L120">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_set_empty_rate</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of empty sets.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

## prediction_set_singleton_rate {#prediction-set-singleton-rate}

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/metrics.py#L135">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">prediction_set_singleton_rate</span>(
    prediction_set,
    *,
    sample_weight=None,
)</code></pre></div>

Fraction of sets containing exactly one label.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">prediction_set</code></dt><dd><p>Boolean membership matrix in classes order.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

<!-- END GENERATED REFERENCE -->
