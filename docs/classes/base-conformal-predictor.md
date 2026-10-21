# BaseConformalPredictor

Read more in the [User Guide](../user-guide/estimators.md).

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L46">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">conformity.</span><span class="api-name">BaseConformalPredictor</span>(
    estimator,
    *,
    auto_calibrate=True,
    calibration_splitter=None,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Wrap a cloned estimator with held-out, unweighted split calibration.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">estimator</code> <span class="api-field-type">BaseEstimator</span></dt><dd><p>Cloneable sklearn estimator; classifiers also require predict_proba.</p></dd>
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Reserve held-out calibration data during fit; defaults to True.</p></dd>
<dt><code class="api-field-name">calibration_splitter</code> <span class="api-field-type">object with split(X, y), or None, default=None</span></dt><dd><p>Splitter producing exactly one (training, calibration) index pair. Defaults to <code>ShuffleSplit(n_splits=1, test_size=0.2)</code>.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">estimator_</code> <span class="api-field-type">BaseEstimator</span></dt><dd><p>Fitted clone; created by fit and removed by parameter changes.</p></dd>
<dt><code class="api-field-name">n_features_in_</code> <span class="api-field-type">int</span></dt><dd><p>Number of validated fitting features.</p></dd>
<dt><code class="api-field-name">feature_names_in_</code> <span class="api-field-type">ndarray</span></dt><dd><p>Input names, present when fitting named string columns.</p></dd>
<dt><code class="api-field-name">classes_</code> <span class="api-field-type">ndarray</span></dt><dd><p>Classifier labels in probability and membership-column order.</p></dd>
<dt><code class="api-field-name">calibration_scores_</code> <span class="api-field-type">ndarray</span></dt><dd><p>Immutable held-out scores; created by successful calibration.</p></dd>
<dt><code class="api-field-name">sorted_calibration_scores_</code> <span class="api-field-type">ndarray</span></dt><dd><p>Immutable sorted scores cached for rank queries.</p></dd>
<dt><code class="api-field-name">n_calibration_</code> <span class="api-field-type">int</span></dt><dd><p>Number of held-out calibration observations.</p></dd>
<dt><code class="api-field-name">alpha_used_</code> <span class="api-field-type">float</span></dt><dd><p>Miscoverage probability from the most recent threshold query.</p></dd>
<dt><code class="api-field-name">quantile_level_</code> <span class="api-field-type">float</span></dt><dd><p>Finite-sample rank divided by calibration count; may exceed one.</p></dd>
<dt><code class="api-field-name">q_level_</code> <span class="api-field-type">float</span></dt><dd><p>Most recent threshold, including infinity when the rank exceeds n.</p></dd>
<dt><code class="api-field-name">is_calibrated_</code> <span class="api-field-type">bool</span></dt><dd><p>Whether held-out scores exist. Refitting clears calibration state.</p></dd>
</dl>

### Examples

```pycon
>>> from sklearn.linear_model import Ridge
>>> from sklearn.model_selection import ShuffleSplit
>>> from conformity import BaseConformalPredictor, ConformalRegressor
>>> issubclass(ConformalRegressor, BaseConformalPredictor)
True
>>> model = ConformalRegressor(Ridge(), calibration_splitter=ShuffleSplit(
...     n_splits=1, test_size=0.25, random_state=0
... ))
>>> model.get_params()["calibration_splitter"].test_size
0.25
>>> model.is_calibrated_
False
```

### Methods

| Method | Purpose |
| --- | --- |
| [calibrate](#calibrate) | Replace calibration scores using data independent of model fitting. |
| [fit](#fit) | Fit a fresh clone; optionally reserve an independent calibration split. |
| [get_metadata_routing](#get-metadata-routing) | Get metadata routing of this object. |
| [get_params](#get-params) | Get parameters for this estimator. |
| [set_fit_request](#set-fit-request) | Request metadata passed to the ``fit`` method. |
| [set_params](#set-params) | Set sklearn parameters and invalidate fitted and calibrated state. |

### calibrate {#calibrate .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L424">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">calibrate</span>(X, y)</code></pre></div>

Replace calibration scores using data independent of model fitting.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,)</span></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>Predictor with replacement calibration scores.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
</dl>

</div>

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L187">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
    X,
    y,
    auto_calibrate=None,
    calibration_splitter=None,
    *,
    sample_weight=None,
    **fit_params,
)</code></pre></div>

Fit a fresh clone; optionally reserve an independent calibration split.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,)</span></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">bool or None, default=None</span></dt><dd><p>Reserve an independent calibration split when fitting.</p></dd>
<dt><code class="api-field-name">calibration_splitter</code> <span class="api-field-type">object with split(X, y), or None, default=None</span></dt><dd><p>Splitter for this fit; requires automatic calibration. A non-None value replaces the constructor splitter. It must produce exactly one training/calibration split. None uses the constructor value, or the default shuffled 80/20 split.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,) or None, default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
<dt><code class="api-field-name">**fit_params</code> <span class="api-field-type">dict</span></dt><dd><p>Estimator-specific fit metadata forwarded to the cloned estimator.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>Fitted predictor; calibrated when automatic calibration is enabled.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">TypeError</code></dt><dd><p>If the estimator cannot be cloned or lacks the required prediction methods.</p></dd>
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

</div>

### get_metadata_routing {#get-metadata-routing .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_metadata_routing">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_metadata_routing</span>()</code></pre></div>

Get metadata routing of this object.

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">routing</code> <span class="api-field-type">MetadataRequest</span></dt><dd><p>A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating routing information.</p></dd>
</dl>

</div>

### get_params {#get-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">get_params</span>(deep=True)</code></pre></div>

Get parameters for this estimator.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">deep</code> <span class="api-field-type">bool, default=True, default=True</span></dt><dd><p>If True, will return the parameters for this estimator and contained subobjects that are estimators.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">params</code> <span class="api-field-type">dict</span></dt><dd><p>Parameter names mapped to their values.</p></dd>
</dl>

</div>

### set_fit_request {#set-fit-request .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_fit_request">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_fit_request</span>(
    *,
    auto_calibrate=&#x27;$UNCHANGED$&#x27;,
    calibration_splitter=&#x27;$UNCHANGED$&#x27;,
    sample_weight=&#x27;$UNCHANGED$&#x27;,
)</code></pre></div>

Request metadata passed to the `fit` method.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>auto_calibrate</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">calibration_splitter</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>calibration_splitter</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>fit</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>

</div>

### set_params {#set-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L138">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_params</span>(**params)</code></pre></div>

Set sklearn parameters and invalidate fitted and calibrated state.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">**params</code> <span class="api-field-type">dict</span></dt><dd><p>Sklearn parameters, including nested estimator parameters.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>Predictor with updated parameters and invalidated fitted state.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
</dl>

</div>

### is_calibrated_

Report whether calibration scores are available.

</div>
<!-- END GENERATED REFERENCE -->
