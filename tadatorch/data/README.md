# tadatorch data

This directory holds the data files tadatorch needs to make predictions.

## Manifest

* `tada.14-0.02.pt`: The trained TADA network weights as a PyTorch state dict. These are the original TADA weights (`tada.14-0.02.hdf5`, as shipped with [TADA_T2](https://github.com/ryanemenecker/TADA_T2)) converted from Keras using `devtools/scripts/convert_keras_weights.py`. The values of the weights are unchanged; they are only re-shaped to match how PyTorch lays out convolution, LSTM and dense layers.
* `scaler_metric.npy`: The metrics used to scale features before they are passed to the network, copied unchanged from TADA_T2. This is a (42, 10) array with one row per feature, where the columns are the fitted attributes of a scikit-learn `StandardScaler` (`mean_`, `var_`, `scale_`, `n_samples_seen_`) followed by those of a `MinMaxScaler` (`min_`, `data_min_`, `data_max_`, `scale_`, `n_samples_seen_`, `data_range_`).
* `five_mer_plddt_sums.npy`: A table of the AlphaFold2 pLDDT scores predicted by [alphaPredict](https://github.com/ryanemenecker/alphaPredict) for every possible 5-residue window (20^5 = 3,200,000 windows), generated using `devtools/scripts/build_plddt_table.py`. The mean predicted pLDDT of each 5-residue window is one of the TADA features, and looking it up here means alphaPredict does not need to be run (or installed) to make predictions. Each entry is the sum of the five per-residue scores in units of 0.0001, stored as a 32-bit integer (alphaPredict rounds its scores to four decimal places, so nothing is lost). The position of a window in the table is the window read as a 5-digit number in base 20, where the digit of each residue is its position in `RKDEQNHSTYCWMAILFVPG`.
* `testing.fasta`: Four 40 amino acid sequences used by the tests, copied unchanged from TADA_T2.
