# ConformalRegressor

Read more in the [User Guide](../user-guide/regression.md).

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L18">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">conformity.</span><span class="api-name">ConformalRegressor</span>(
    estimator,
    *,
    auto_calibrate=True,
    tts_kwargs=None,
    interval_method=&#x27;symmetric&#x27;,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Split-conformal prediction intervals for single-output regression.

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">estimator</code> <span class="api-field-type">BaseEstimator</span></dt><dd><p>Cloneable sklearn regressor or preprocessing-and-regression pipeline.</p></dd>
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>Reserve held-out calibration data during fit; defaults to True.</p></dd>
<dt><code class="api-field-name">tts_kwargs</code> <span class="api-field-type">TrainTestSplitKwargs or None, default=None</span></dt><dd><p>train_test_split options for automatic calibration. By default, test_size=0.2, shuffle=True and stratify=None.</p></dd>
<dt><code class="api-field-name">interval_method</code> <span class="api-field-type">{&quot;symmetric&quot;, &quot;asymmetric&quot;}, default=&#x27;symmetric&#x27;</span></dt><dd><p>Symmetric absolute-residual calibration (default), or separate signed residual quantiles with alpha / 2 allocated to each tail.</p></dd>
</dl>

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">lower_q_level_</code> <span class="api-field-type">float</span></dt><dd><p>Most recent lower correction for asymmetric intervals.</p></dd>
<dt><code class="api-field-name">upper_q_level_</code> <span class="api-field-type">float</span></dt><dd><p>Most recent upper correction for asymmetric intervals.</p></dd>
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
>>> from sklearn.datasets import make_regression
>>> from sklearn.linear_model import Ridge
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from conformity import ConformalRegressor
>>> X, y = make_regression(
...     n_samples=120, n_features=3, noise=4, random_state=0
... )
>>> X_dev, X_test, y_dev, y_test = train_test_split(
...     X, y, test_size=0.25, random_state=1
... )
>>> X_train, X_calib, y_train, y_calib = train_test_split(
...     X_dev, y_dev, test_size=1 / 3, random_state=2
... )
>>> model = ConformalRegressor(
...     Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
...     auto_calibrate=False,
... )
>>> _ = model.fit(X_train, y_train)
>>> _ = model.calibrate(X_calib, y_calib)
>>> points, intervals = model.predict(X_test, alpha=0.1, return_interval=True)
>>> points.shape, intervals.shape
((30,), (30, 2))
>>> model.n_calibration_
30
```

### Methods

| Method | Purpose |
| --- | --- |
| [calibrate](#calibrate) | Replace held-out scores and return this predictor for method chaining. |
| [evaluate](#evaluate) | Return point error, coverage, tail misses, width and interval score. |
| [fit](#fit) | Fit a fresh clone; optionally reserve an independent calibration split. |
| [get_metadata_routing](#get-metadata-routing) | Get metadata routing of this object. |
| [get_params](#get-params) | Get parameters for this estimator. |
| [predict](#predict) | Return points or (points, intervals). |
| [predict_interval](#predict-interval) | Return [lower, upper] bounds with shape (n_samples, 2). |
| [predict_point](#predict-point) | Return point predictions without requiring calibration. |
| [score](#score) | Return point-prediction R², using point predictions. |
| [set_fit_request](#set-fit-request) | Request metadata passed to the ``fit`` method. |
| [set_params](#set-params) | Set sklearn parameters and invalidate fitted and calibrated state. |
| [set_predict_request](#set-predict-request) | Request metadata passed to the ``predict`` method. |
| [set_score_request](#set-score-request) | Request metadata passed to the ``score`` method. |
### calibrate {#calibrate .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L128">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">calibrate</span>(X, y)</code></pre></div>

Replace held-out scores and return this predictor for method chaining.

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

### evaluate {#evaluate .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L325">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">evaluate</span>(
    X,
    y,
    alpha=0.05,
    *,
    sample_weight=None,
)</code></pre></div>

Return point error, coverage, tail misses, width and interval score.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,)</span></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">float, default=0.05</span></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,) or None, default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">report</code> <span class="api-field-type">dict of str to float</span></dt><dd><p>Keys are r2, mae, mse, coverage, lower_miss_rate, upper_miss_rate, mean_width, normalized_mean_width and interval_score. Normalized width divides mean width by the unweighted test-target IQR and is infinite when that IQR is zero. Tail miss rates are empirical diagnostics; for asymmetric intervals compare each with alpha / 2, allowing for finite test-sample variation.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
<dt><code class="api-field-name">RuntimeError</code></dt><dd><p>If conformal output is requested before calibration.</p></dd>
</dl>

</div>

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L186">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
    X,
    y,
    auto_calibrate=None,
    tts_kwargs=None,
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
<dt><code class="api-field-name">tts_kwargs</code> <span class="api-field-type">TrainTestSplitKwargs or None, default=None</span></dt><dd><p>train_test_split options for this fit; requires automatic calibration. If provided, this mapping replaces the constructor mapping. Missing options use defaults: test_size=0.2, train_size=None, random_state=None, shuffle=True and stratify=None. Set <code>random_state</code> here for reproducible automatic splits.</p></dd>
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

### predict {#predict .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L251">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict</span>(
    X,
    alpha=0.05,
    *,
    return_interval=False,
)</code></pre></div>

Return points or (points, intervals).

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">float, default=0.05</span></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">return_interval</code> <span class="api-field-type">bool, default=False</span></dt><dd><p>Request conformal output alongside ordinary predictions.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">predictions</code> <span class="api-field-type">ndarray or tuple of ndarray</span></dt><dd><p>Point predictions, or (points, intervals) when return_interval=True.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
<dt><code class="api-field-name">RuntimeError</code></dt><dd><p>If conformal output is requested before calibration.</p></dd>
</dl>

</div>

### predict_interval {#predict-interval .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L221">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_interval</span>(X, alpha=0.05)</code></pre></div>

Return [lower, upper] bounds with shape (n_samples, 2).

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">float, default=0.05</span></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">intervals</code> <span class="api-field-type">ndarray of shape (n_samples, 2)</span></dt><dd><p>Closed lower and upper bounds; may be outward infinite.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
<dt><code class="api-field-name">RuntimeError</code></dt><dd><p>If conformal output is requested before calibration.</p></dd>
</dl>

</div>

### predict_point {#predict-point .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L198">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_point</span>(X)</code></pre></div>

Return point predictions without requiring calibration.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">predictions</code> <span class="api-field-type">ndarray of shape (n_samples,)</span></dt><dd><p>Point predictions from the fitted estimator.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
</dl>

</div>

### score {#score .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/regressor.py#L296">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">score</span>(
    X,
    y,
    sample_weight=None,
)</code></pre></div>

Return point-prediction R², using point predictions.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code> <span class="api-field-type">{array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)</span></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code> <span class="api-field-type">array-like of shape (n_samples,)</span></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">array-like of shape (n_samples,) or None, default=None</span></dt><dd><p>Optional finite, nonnegative weights; fitting or evaluation only, never calibration.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">score</code> <span class="api-field-type">float</span></dt><dd><p>Point-prediction R² for regression or accuracy for classification.</p></dd>
</dl>

#### Raises

<dl class="api-fields">
<dt><code class="api-field-name">ValueError</code></dt><dd><p>If input shapes, values, weights or configuration are invalid.</p></dd>
<dt><code class="api-field-name">NotFittedError</code></dt><dd><p>If fit has not established a fitted estimator.</p></dd>
</dl>

</div>

### set_fit_request {#set-fit-request .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_fit_request">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_fit_request</span>(
    *,
    auto_calibrate=&#x27;$UNCHANGED$&#x27;,
    sample_weight=&#x27;$UNCHANGED$&#x27;,
    tts_kwargs=&#x27;$UNCHANGED$&#x27;,
)</code></pre></div>

Request metadata passed to the `fit` method.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>auto_calibrate</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">tts_kwargs</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>tts_kwargs</code> parameter in <code>fit</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>

</div>

### set_params {#set-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L137">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_params</span>(**params)</code></pre></div>

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

### set_predict_request {#set-predict-request .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_predict_request">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_predict_request</span>(
    *,
    alpha=&#x27;$UNCHANGED$&#x27;,
    return_interval=&#x27;$UNCHANGED$&#x27;,
)</code></pre></div>

Request metadata passed to the `predict` method.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>alpha</code> parameter in <code>predict</code>.</p></dd>
<dt><code class="api-field-name">return_interval</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>return_interval</code> parameter in <code>predict</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>

</div>

### set_score_request {#set-score-request .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_score_request">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_score_request</span>(*, sample_weight=&#x27;$UNCHANGED$&#x27;)</code></pre></div>

Request metadata passed to the `score` method.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED, default=&#x27;$UNCHANGED$&#x27;</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>score</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>

</div>

### is_calibrated_

Report whether calibration scores are available.

</div>
<!-- END GENERATED REFERENCE -->
