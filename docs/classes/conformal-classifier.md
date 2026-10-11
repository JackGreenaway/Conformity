# ConformalClassifier

<!-- BEGIN GENERATED REFERENCE -->
## Reference

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L17">[source]</a><pre><code><span class="api-keyword">class </span><span class="api-module">conformity.</span><span class="api-name">ConformalClassifier</span>(
    estimator,
    *,
    method=&#x27;lac&#x27;,
    auto_calibrate=True,
    tts_kwargs=None,
    calibration_size=0.2,
    random_state=None,
)</code></pre></div>

<div class="api-class-body" markdown="1">

Split conformal classification using LAC or deterministic APS scores.

### Attributes

<dl class="api-fields">
<dt><code class="api-field-name">method_</code> <span class="api-field-type">{&quot;lac&quot;, &quot;aps&quot;}</span></dt><dd><p>Score method recorded by successful calibration; changing it requires recalibration. Shared fitted attributes are documented on the base class.</p></dd>
</dl>

### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">estimator</code></dt><dd><p>Cloneable sklearn classifier implementing predict_proba.</p></dd>
<dt><code class="api-field-name">method</code></dt><dd><p>Use lac (1 - probability) or aps (cumulative ranked probability).</p></dd>
<dt><code class="api-field-name">auto_calibrate</code></dt><dd><p>Reserve held-out calibration data during fit; defaults to True.</p></dd>
<dt><code class="api-field-name">tts_kwargs</code></dt><dd><p>Optional train_test_split overrides, applied only with auto_calibrate.</p></dd>
<dt><code class="api-field-name">calibration_size</code></dt><dd><p>Calibration fraction or sample count; defaults to 0.2.</p></dd>
<dt><code class="api-field-name">random_state</code></dt><dd><p>Seed or RandomState for reproducible automatic splits.</p></dd>
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

### Methods

| Method | Purpose |
| --- | --- |
| [calibrate](#calibrate) | Replace held-out scores and return this predictor for method chaining. |
| [evaluate](#evaluate) | Return accuracy, log loss, coverage and prediction set diagnostics. |
| [fit](#fit) | Fit a fresh clone; optionally reserve an independent calibration split. |
| [get_metadata_routing](#get-metadata-routing) | Get metadata routing of this object. |
| [get_params](#get-params) | Get parameters for this estimator. |
| [predict](#predict) | Return point labels, or (boolean membership sets, probabilities). |
| [predict_p_values](#predict-p-values) | Return conservative conformal p-values, including calibration ties. |
| [predict_point](#predict-point) | Return the wrapped estimator's labels without calibration. |
| [predict_proba](#predict-proba) | Return validated probabilities in classes_ order. |
| [predict_set](#predict-set) | Return a boolean membership matrix of shape (n_samples, n_classes). |
| [score](#score) | Return point accuracy using point predictions. |
| [set_fit_request](#set-fit-request) | Request metadata passed to the ``fit`` method. |
| [set_params](#set-params) | Set sklearn parameters and invalidate fitted and calibrated state. |
| [set_predict_request](#set-predict-request) | Request metadata passed to the ``predict`` method. |
| [set_score_request](#set-score-request) | Request metadata passed to the ``score`` method. |
### calibrate {#calibrate .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L107">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">calibrate</span>(X, y)</code></pre></div>

Replace held-out scores and return this predictor for method chaining.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
</dl>

</div>

### evaluate {#evaluate .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L236">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">evaluate</span>(
    X,
    y,
    alpha=0.05,
    *,
    sample_weight=None,
)</code></pre></div>

Return accuracy, log loss, coverage and prediction set diagnostics.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">alpha</code></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
</dl>

</div>

### fit {#fit .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L156">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">fit</span>(
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
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">auto_calibrate</code></dt><dd><p>Reserve an independent calibration split when fitting.</p></dd>
<dt><code class="api-field-name">tts_kwargs</code></dt><dd><p>Optional train_test_split overrides; requires automatic calibration.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
<dt><code class="api-field-name">fit_params</code></dt><dd><p>Estimator-specific fit metadata forwarded to the cloned estimator.</p></dd>
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
<dt><code class="api-field-name">deep</code> <span class="api-field-type">bool, default=True</span></dt><dd><p>If True, will return the parameters for this estimator and contained subobjects that are estimators.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">params</code> <span class="api-field-type">dict</span></dt><dd><p>Parameter names mapped to their values.</p></dd>
</dl>

</div>

### predict {#predict .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L192">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict</span>(
    X,
    alpha=0.05,
    *,
    return_set=False,
)</code></pre></div>

Return point labels, or (boolean membership sets, probabilities).

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">alpha</code></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
<dt><code class="api-field-name">return_set</code></dt><dd><p>Request conformal output alongside ordinary predictions.</p></dd>
</dl>

</div>

### predict_p_values {#predict-p-values .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L172">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_p_values</span>(X)</code></pre></div>

Return conservative conformal p-values, including calibration ties.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
</dl>

</div>

### predict_point {#predict-point .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L137">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_point</span>(X)</code></pre></div>

Return the wrapped estimator's labels without calibration.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
</dl>

</div>

### predict_proba {#predict-proba .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L148">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_proba</span>(X)</code></pre></div>

Return validated probabilities in classes_ order.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
</dl>

</div>

### predict_set {#predict-set .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L158">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">predict_set</span>(X, alpha=0.05)</code></pre></div>

Return a boolean membership matrix of shape (n_samples, n_classes).

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">alpha</code></dt><dd><p>Miscoverage probability strictly between zero and one.</p></dd>
</dl>

</div>

### score {#score .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/classifier.py#L220">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">score</span>(
    X,
    y,
    sample_weight=None,
)</code></pre></div>

Return point accuracy using point predictions.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">X</code></dt><dd><p>Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.</p></dd>
<dt><code class="api-field-name">y</code></dt><dd><p>One target per sample; classification labels follow the fitted classes.</p></dd>
<dt><code class="api-field-name">sample_weight</code></dt><dd><p>Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.</p></dd>
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
<dt><code class="api-field-name">auto_calibrate</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>auto_calibrate</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>fit</code>.</p></dd>
<dt><code class="api-field-name">tts_kwargs</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>tts_kwargs</code> parameter in <code>fit</code>.</p></dd>
</dl>

#### Returns

<dl class="api-fields">
<dt><code class="api-field-name">self</code> <span class="api-field-type">object</span></dt><dd><p>The updated object.</p></dd>
</dl>

</div>

### set_params {#set-params .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://github.com/JackGreenaway/Conformity/blob/main/src/conformity/base.py#L118">[source]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_params</span>(**params)</code></pre></div>

Set sklearn parameters and invalidate fitted and calibrated state.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">params</code></dt><dd><p>Sklearn parameters, including nested estimator parameters.</p></dd>
</dl>

</div>

### set_predict_request {#set-predict-request .api-method-title}

<div class="api-method-body" markdown="1">

<div class="api-signature"><a class="api-source" href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.set_predict_request">[inherited API]</a><pre><code><span class="api-keyword"></span><span class="api-module"></span><span class="api-name">set_predict_request</span>(
    *,
    alpha=&#x27;$UNCHANGED$&#x27;,
    return_set=&#x27;$UNCHANGED$&#x27;,
)</code></pre></div>

Request metadata passed to the `predict` method.

#### Parameters

<dl class="api-fields">
<dt><code class="api-field-name">alpha</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>alpha</code> parameter in <code>predict</code>.</p></dd>
<dt><code class="api-field-name">return_set</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>return_set</code> parameter in <code>predict</code>.</p></dd>
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
<dt><code class="api-field-name">sample_weight</code> <span class="api-field-type">str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED</span></dt><dd><p>Metadata routing for <code>sample_weight</code> parameter in <code>score</code>.</p></dd>
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
