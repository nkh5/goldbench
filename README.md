# GoldBench

A reproducible benchmark testing whether model complexity helps when forecasting
gold futures (GC=F) on daily data. Five model families are compared across two
targets, one essentially unforecastable (next day direction) and one genuinely
forecastable (short horizon volatility), under leakage controlled combinatorial
purged cross validation. The project then compresses the trained models for edge
deployment and measures the size, speed, and accuracy tradeoff.

**Central finding:** on low signal financial data, simple models match or beat
complex ones on both targets. The real skill is knowing which questions are
answerable.

## Why this project exists

Most "predict the market with an LSTM" projects report a high accuracy number that
quietly comes from data leakage or an unreported baseline. GoldBench is built the
opposite way: the evaluation infrastructure is the centerpiece, and the results are
reported honestly even when they are null. The question is not "can I beat gold,"
it is "can I build an evaluation that would detect signal if it existed, and report
statistically humble results if it did not."

## Results

### Direction track: predict next day up or down

Five models predict whether tomorrow's close will be higher than today's, scored by
AUC (the primary metric, since it is unaffected by class imbalance) across 28
combinatorial purged cross validation splits.

| Model | AUC (mean) | AUC std | Accuracy | Brier | Note |
|---|---|---|---|---|---|
| naive | 0.500 | 0.000 | 0.558 | 0.247 | floor: always predicts the base rate |
| logistic | 0.515 | 0.024 | 0.543 | 0.251 | only model above random AUC |
| LightGBM | 0.484 | 0.029 | 0.548 | 0.249 | overfit below random (21 of 28 folds under 0.50) |
| LSTM | 0.494 | 0.015 | 0.537 | 0.287 | near random, lowest variance |
| PatchTST | 0.506 | 0.018 | 0.494 | 0.332 | target only (channel independent), barely above random |

**Reading it:** the simplest model that uses features (logistic regression) is the
only one that meaningfully beats the random baseline. Both high capacity models
(LightGBM and the LSTM) land at or below a coin flip out of sample, and the modern
transformer barely clears random from price action alone. The naive baseline is
right 55.8 percent of the time simply because gold rose over this period, so any
model bragging about "54 percent accuracy" is in fact losing to a model that does no
work. This is the bias variance tradeoff and weak form market efficiency, measured
directly rather than assumed.

### Volatility track: predict short horizon realized volatility

The same model families (plus a classical GARCH baseline) predict next period
realized volatility, a continuous target. Because volatility clusters, this target
is genuinely forecastable, unlike direction. Scored by rank correlation and QLIKE
(the volatility literature's preferred metrics); R squared is omitted deliberately
(see Methodology).

| Model | Correlation (mean) | RMSE | QLIKE | Note |
|---|---|---|---|---|
| persistence | 0.198 | 0.112 | 32.07 | naive baseline: tomorrow looks like recently |
| GARCH | 0.267 | 0.098 | -2.65 | classical model, clearly beats persistence |
| ridge | 0.276 | 0.095 | -2.35 | best on every metric |
| LightGBM | 0.123 | 0.106 | -2.16 | overfit, same pattern as direction track |
| LSTM | unstable | varies | large | median fold correlation 0.000; 6 of 28 folds negative |

**Reading it:** here the models work. Ridge regression (the simple linear model
again) wins, narrowly beating the classical GARCH model, and both clearly beat the
naive persistence baseline. The pattern from the direction track repeats: the high
capacity models fail, with LightGBM overfitting and the LSTM unstable (it scored a
strong 0.467 correlation in one fold but a median of zero across all 28, evidence of
an overparameterized model on insufficient data). PatchTST was attempted and cut: it
produced inverted forecasts on this target.

### Edge deployment: model compression

The best models were exported to ONNX and quantized to INT8, measuring the tradeoff
that drives on device deployment.

| Experiment | Size | Latency | Accuracy | Insight |
|---|---|---|---|---|
| LightGBM to ONNX | 34 percent smaller | 2.8x faster | identical predictions | ONNX runtime is faster than native LightGBM |
| LightGBM INT8 quantization | unchanged | unchanged | identical (0.00000 shift) | trees are structure, not weights, so quantization does nothing |
| LSTM INT8 quantization | 32 percent smaller | no CPU speedup | AUC change -0.0009 | neural nets are weight heavy, so quantization compresses; speed benefit is hardware dependent |

**Reading it:** quantization compressed the weight heavy LSTM by roughly a third with
negligible accuracy loss (AUC moved 0.0009), but left the tree model byte for byte
identical because trees encode decision rules rather than multipliable weights. The
LSTM saw no speed benefit on CPU, because ONNX runtime lacks INT8 optimized kernels
for recurrent operations; on hardware with dedicated INT8 acceleration the result
would differ. Measuring this tradeoff across two architectures, rather than assuming
quantization always helps, is the point.

## Methodology

**Combinatorial purged cross validation.** Standard k fold leaks future information
into the past on time series. GoldBench partitions the data into 8 contiguous blocks
and tests on every pair (28 splits, 7 reconstructable paths), with two leakage
controls: purging removes training rows whose forward looking labels overlap a test
block, and a 5 day embargo drops a buffer on each side so no rolling feature window
straddles the boundary. Every result is reported as a distribution across the 28
splits, not a single point estimate.

**Leakage discipline.** Every scaler is fit inside the training fold only (via
sklearn Pipelines), every rolling feature uses only past data, and a unit test
poisons all future rows with NaN, recomputes features, and asserts past features are
unchanged. The cross validation purge widens automatically with the label's forward
horizon.

**Metric choice for volatility.** R squared is omitted from the volatility results on
purpose. It grades against each fold's shifting mean, punishes conservatively scaled
forecasts even when their timing is perfect, and is dominated by the rare extreme
days of a fat tailed, zero bounded target. The volatility literature uses QLIKE and
correlation instead, so GoldBench leads with those. Knowing why R squared misleads
here is part of the project.

**Tier appropriate features.** Linear and tree models receive engineered scalar
features (lagged returns, indicator transforms, rolling statistics); the sequence
models receive 60 day windows of raw channels. PatchTST is channel independent by
architecture and accepts no exogenous features, so it forecasts from the target's own
history alone, a property disclosed rather than worked around.

## Limitations

This project uses roughly 1,150 daily observations on a single instrument, which is
small by machine learning standards and severely small for deep learning. Several
honest constraints follow.

Single asset daily data cannot prove alpha. With this sample size, even a positive
result would not survive the multiple testing corrections that the literature
requires for financial strategies.

The LSTM on the direction track was trained as regression with isotonic calibration,
because the neuralforecast library (v3.1.5) does not support Bernoulli loss on
recurrent models. This produced its poor Brier scores and is a documented workaround,
not a hidden choice.

The LSTM on the volatility track was unstable across folds (median correlation zero
despite a strong single fold), which is itself evidence of overfitting on limited
data, the project's recurring theme.

The volatility target uses a short forward window, so consecutive targets overlap;
this is handled through the widened purge but introduces mild autocorrelation in the
target series.

No live trading strategy or transaction cost backtest is included; that is future
work, not a claim being made here.

## Reproducing

```bash
uv sync
uv pip install -e .
uv run python -m src.features          # build direction features
uv run python -m src.features_vol      # build volatility features
uv run pytest tests/ -v                # leakage and cross validation tests
uv run python -m src.baseline_runner   # direction track (5 models)
uv run python -m src.vol_runner        # volatility track (5 models)
uv run python -m src.edge.export_onnx  # ONNX export benchmark
uv run python -m src.edge.quant_lstm   # LSTM quantization benchmark
```

## References

* Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?"
* Nie et al., ICLR 2023, "A Time Series is Worth 64 Words" (PatchTST)
* Lopez de Prado, 2018, "Advances in Financial Machine Learning" (purged cross validation, ch. 7 and 12)
* Engle, 2003 Nobel lecture (ARCH and GARCH volatility modeling)
* Patton, 2011, on robust loss functions for volatility forecasting (QLIKE)

## What I learned

The headline lesson is that on noisy financial data, model capacity is a liability:
across both a classification target and a regression target, the simplest models
matched or beat gradient boosting and neural networks, and the complex models either
overfit below random or destabilized. The second lesson is that choosing the right
metric for the data type (AUC over accuracy under class imbalance, QLIKE over R
squared for volatility) changes the conclusion more than choosing the right model.
The third is that compression techniques are architecture dependent: quantization
that does nothing to a tree meaningfully shrinks a neural net.