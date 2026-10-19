# Visual gallery

Use this gallery for a quick visual introduction to split calibration, classification scores, regression intervals and set diagnostics. Select a figure to open it at full size. Each chart uses held-out synthetic data; the diagrams explain the workflow and score rules.

## How calibration works

<div class="visual-gallery">
<article class="visual-card" markdown="1">

### Split before fitting

For the usual split-conformal guarantee, hold calibration rows out before fitting any learned preprocessing or model. The wrapper can do this automatically when it encloses the complete pipeline.

[![Workflow diagram showing a fit partition and held-out calibration partition feeding calibration before future prediction](../images/calibration-workflow.svg)](../images/calibration-workflow.svg)

</article>
<article class="visual-card" markdown="1">

### Scores determine set membership

LAC and deterministic APS assign candidate scores differently. A calibrated threshold selects candidates whose score is at or below that threshold. This example uses illustrative thresholds to show the rule.

[![Table diagram comparing LAC and deterministic APS scores for three labels and showing which labels meet illustrative thresholds](../images/classification-score-threshold.svg)](../images/classification-score-threshold.svg)

</article>
</div>

## What the prediction regions look like

<div class="visual-gallery visual-gallery--plots">
<article class="visual-card" markdown="1">

### Symmetric and asymmetric regression intervals

Both panels use the same training, calibration and test partitions. The asymmetric method can use different lower and upper corrections. Neither method changes its correction from row to row.

[![Two vertically stacked scatter plots comparing symmetric and asymmetric 90 percent conformal regression intervals on held-out delivery outcomes](../images/symmetric-asymmetric-intervals.png)](../images/symmetric-asymmetric-intervals.png)

</article>
<article class="visual-card" markdown="1">

### Classification set size and empirical coverage

Point size in the feature plot indicates the number of labels in each prediction set. The other panels show set-size counts and observed coverage by true class; class-wise bars are descriptive and do not imply class-conditional guarantees.

[![Three vertically stacked panels showing held-out classification samples sized by set cardinality, set size counts, and empirical coverage by class](../images/classification-prediction-sets.png)](../images/classification-prediction-sets.png)

</article>
</div>

## Run the visual examples

The diagrams are static. Reproduce the data-driven figures with the [downloadable plotting script](plots.md), or follow the [regression](regression.md) and [classification](classification.md) workflows. Matplotlib is optional; see [installation](../installation.md).
