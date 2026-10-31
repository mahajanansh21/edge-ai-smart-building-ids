# Methodology notes

1. Acquire and document the public dataset and its usage terms.
2. Inspect real file names, columns, target labels, class counts and data quality.
3. Investigate identifiers, timestamps and other possible leakage fields.
4. Create a justified train/validation/test split before fitting transformations.
5. Fit all preprocessing only on training data.
6. Compare baseline classifiers on identical splits.
7. Select a model using detection quality and edge-resource measurements.
8. Evaluate once on the held-out test set and retain the full experiment settings.
9. Benchmark local CPU inference and clearly label the laptop as an edge simulation.
