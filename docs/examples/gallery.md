# Examples gallery

Browse the visual examples below for a quick introduction to split calibration, classification scores, regression intervals and set diagnostics. Select an image to open it at full size. The diagrams explain the workflow and score rules; the plots use held-out synthetic data.

## How calibration works

<div class="visual-gallery">
  <article class="visual-card">
    <h3>Split before fitting</h3>
    <p>For the usual split-conformal guarantee, hold calibration rows out before fitting learned preprocessing or a model. Automatic splitting handles this when the wrapper encloses the complete pipeline.</p>
    <a href="../images/calibration-workflow.svg"><img src="../images/calibration-workflow.svg" alt="Workflow diagram: development data is split into a fit partition and held-out calibration partition; the full pipeline fits only on fit rows, calibration scores set a rank threshold, and the threshold is used for future prediction."></a>
  </article>
  <article class="visual-card">
    <h3>Scores determine set membership</h3>
    <p>LAC and deterministic APS assign candidate scores differently. A calibrated threshold selects candidates whose score is at or below it. This example uses illustrative thresholds.</p>
    <a href="../images/classification-score-threshold.svg"><img src="../images/classification-score-threshold.svg" alt="Comparison of LAC scores 1 minus probability and deterministic APS cumulative scores for labels A, B and C, with illustrative thresholds that include A for LAC and A plus B for APS."></a>
  </article>
</div>

## What prediction regions look like

<div class="visual-gallery visual-gallery--plots">
  <article class="visual-card">
    <h3>Symmetric and asymmetric regression intervals</h3>
    <p>Both panels use the same training, calibration and test partitions. The asymmetric method can use different lower and upper corrections; neither method changes its correction from row to row.</p>
    <a href="../images/symmetric-asymmetric-intervals.png"><img src="../images/symmetric-asymmetric-intervals.png" alt="Two vertically stacked scatter plots comparing symmetric and asymmetric 90 percent conformal regression intervals on held-out delivery outcomes."></a>
  </article>
  <article class="visual-card">
    <h3>Classification set size and empirical coverage</h3>
    <p>Point size in the feature plot indicates the number of labels in each prediction set. The other panels show set-size counts and observed coverage by true class. Class-wise bars are descriptive and do not imply class-conditional guarantees.</p>
    <a href="../images/classification-prediction-sets.png"><img src="../images/classification-prediction-sets.png" alt="Three vertically stacked panels show held-out classification samples sized by set cardinality, prediction set size counts, and empirical coverage by class."></a>
  </article>
</div>

## Run the visual examples

The diagrams are static. Reproduce the data-driven figures with the [downloadable plotting script](plots.md), or follow the [regression](regression.md) and [classification](classification.md) workflows. Matplotlib is optional; see [installation](../installation.md). For the complete list of hands-on examples, see the [runnable examples index](index.md).
