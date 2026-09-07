# Reading the experiments

The experiments begin with the initial objective comparisons, using an earlier
trainer. Subsequent prompt calibration examines the existing model, and the
response study establishes how a complete answer should end. Once EOS supervision and joint response fitting are fixed, the proposal studies
ask whether supplying the known answer helps discover useful candidates within
each approximation.

The common grid then varies generation, question batch size and repeated
fitting. Its equal-weight control separates sampling alone from using the
answer likelihood to allocate the update. The frozen answer scorer and ESS
floor are later responses to the observed reuse problem, rather than original
predictions. Confirmation studies evaluate the selected configurations on fresh
training seeds, followed by comparisons with established methods and transfer
without further training.

[The experiment index](experiments/EXPERIMENTS.md) identifies each original
version and its configuration. [The frozen numeric exports](experiments/results)
retain the comparisons, including results that did not improve training. The
current APIs also retain a few later exploratory update rules for compatibility;
they are not additional evidence for the thesis's principal delta algorithm.
