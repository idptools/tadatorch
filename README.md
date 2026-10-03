tadatorch
==============================

tadatorch predicts transcriptional activation domains (ADs) in protein sequences. It is a PyTorch re-implementation of TADA ([Morffy, Van den Broeck et al. 2024, Nature](https://doi.org/10.1038/s41586-024-07707-3)), a neural network trained on the activation measured for tens of thousands of 40-amino-acid protein fragments in yeast.

Give tadatorch a sequence (or a FASTA file), and it will return a TAD score between 0 and 1 for every 40-amino-acid window, where higher scores mean the window is more likely to activate transcription. It can also turn those scores into annotated activation domains, with start and end positions, from Python or from the command line.

tadatorch uses the same trained network as [TADA_T2](https://github.com/ryanemenecker/TADA_T2) (the TensorFlow2 version of TADA) and its user-facing functions take the same arguments and return the same format, but it:

* does not need TensorFlow, localcider, or alphaPredict;
* is around 85 times faster (0.35 milliseconds per window on one CPU core);
* annotates activation domains, rather than only scoring windows;
* gives scores that are very close to, but not identical to, those from TADA_T2 (see [Accuracy and how tadatorch relates to TADA and TADA_T2](#accuracy-and-how-tadatorch-relates-to-tada-and-tada_t2)).

## Contents

* [Installation](#installation)
* [Quick start](#quick-start)
* [What a TAD score means](#what-a-tad-score-means)
* [Sequences tadatorch can score](#sequences-tadatorch-can-score)
* [Python interface](#python-interface)
* [Command line interface](#command-line-interface)
* [How activation domains are annotated](#how-activation-domains-are-annotated)
* [How tadatorch works](#how-tadatorch-works)
* [Accuracy and how tadatorch relates to TADA and TADA_T2](#accuracy-and-how-tadatorch-relates-to-tada-and-tada_t2)
* [Speed](#speed)
* [Package layout](#package-layout)
* [Development](#development)
* [Citation](#citation)
* [License and credits](#license-and-credits)

# Installation

tadatorch needs Python 3.10 or later. It can be installed from a local copy of this repository. From the directory holding this README, run:

```bash
pip install -e .
```

The dependencies are installed automatically:

* [PyTorch](https://pytorch.org/)
* numpy
* [protfasta](https://github.com/idptools/protfasta)
* [sparrow](https://github.com/idptools/sparrow), installed from GitHub. Note that the package called `sparrow` on PyPI is unrelated, so `pip install sparrow` will not work. tadatorch needs a version of sparrow that includes `sparrow.patterning.vectorized`.

Two groups of optional dependencies are only needed for development: `pip install -e ".[test]"` adds pytest, and `pip install -e ".[devtools]"` adds h5py and alphaPredict, which are only used by the scripts that generated the data files shipped with tadatorch (see [Package layout](#package-layout)).

To check the installation, run the tests (this takes a few seconds):

```bash
pytest
```

# Quick start

From Python:

```python
from tadatorch import predict, predict_activation_domains

# a TAD score for a 40 amino acid sequence
predict("EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD")
# {'EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD': [['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.6482064127922058]]}

# the activation domains in a whole protein (yeast Gcn4)
domains = predict_activation_domains(gcn4_sequence)
# [ActivationDomain(start=40, end=151, ...)]

# tighter domains, using only the central residues of each window
domains = predict_activation_domains(gcn4_sequence, method="central_residues")
# [ActivationDomain(start=70, end=132, ...)]
```

From the command line:

```bash
tadatorch proteins.fasta -o activation_domains.tsv
tadatorch proteins.fasta -o activation_domains.tsv --method central_residues
```

# What a TAD score means

TAD scores range between 0 and 1, where lower values are predicted to be less likely to be TADs and higher values are predicted to be more likely to be TADs. Note that **the values are predictions and should be treated as such**. Nothing can substitute for experimental validation; however, TAD scores can guide new hypotheses or experimental planning.

In practice, the scores cover a narrow range: across every 40-amino-acid window of 164 yeast transcription factors, they run from 0.13 to 0.72, and the median is 0.19. They are therefore best treated as rankings rather than probabilities.

### Is there a threshold above which something should be considered a TAD?

In the original publication, sequences with **scores above 0.5** were classified as TADs, and tadatorch uses 0.5 as the default threshold. This is a good starting point, but *it is an arbitrary threshold, and the scores should be used as a guide, not a definitive answer*.

As a guide to how the threshold behaves, tadatorch was compared with the activation measured for 7,370 tiles (53 amino acids long) of 164 yeast transcription factors by [Sanborn et al. 2021](https://doi.org/10.7554/eLife.68068), none of which were used to train TADA. Each tile was given the highest score of the 40 amino acid windows inside it:

* An active tile received a higher score than an inactive tile 95% of the time (ROC AUC of 0.948), and the average precision was 0.65 (0.07 would be expected by chance).
* At a threshold of 0.5, 65% of active tiles were called, and 55% of called tiles were active.
* Raising the threshold to 0.6 calls fewer tiles, but more of them are active (in the same comparison, around 80% of tiles called at 0.6 were active, but only around 30% of active tiles were called).

### How were activation domains originally identified?

To simplify things a bit, the activation domains TADA was trained on were identified using a high-throughput assay in *Saccharomyces cerevisiae* that measured the ability of tens of thousands of 40-amino-acid fragments (from *Arabidopsis* transcription factors) to drive transcription of a reporter. For more information, please see the [publication](https://doi.org/10.1038/s41586-024-07707-3).

# Sequences tadatorch can score

**Sequences must be made up of the 20 standard amino acids, in uppercase.** Anything else raises an exception because some features are calculated by matching residues against uppercase one-letter codes. When sequences are read from a FASTA file (with ``predict_from_fasta`` or the command line tool), non-standard residues are converted first: B to N, U to C, X to G and Z to Q, with '*', '-' and spaces removed.

**TADA scores 40 amino acid sequences.** For longer sequences, tadatorch uses a sliding window: by default, it scores every 40-amino-acid window, moving one residue at a time. Note that **this is equivalent to breaking your sequence into 'chunks' of 40 amino acids and scoring each one on its own**, so each score only takes in information from 40 amino acids.

**Sequences shorter than 40 amino acids** can be padded with randomly chosen residues to make them 40 amino acids long, either evenly on both sides or only at the N- or C-terminus, using only G and S or a random selection of all 20 amino acids. **This is NOT ideal**, and we recommend that you do not predict TAD scores for sequences shorter than 40 amino acids, as the predictor was not made for this and the predictions may not be accurate. By default, this is not allowed, and you must set ``safe_mode=False`` to use it. ``predict_activation_domains`` and the command-line tool do not pad sequences and skip (or raise an exception for) sequences under 40 amino acids.

# Python interface

The main functions are imported from the top of the package:

```python
from tadatorch import predict, predict_from_fasta, predict_activation_domains
```

``predict`` and ``predict_from_fasta`` return TAD scores, and take the same arguments and return the same format as the functions of the same names in TADA_T2. If you are moving from TADA_T2, the only change needed is the import (which was previously `from TADA_T2.TADA import predict, predict_from_fasta`). ``predict_activation_domains`` annotates activation domains.

## predict

``predict`` returns TAD scores for a sequence or a list of sequences.

```python
predict(sequences)
```

**Parameters**:

* ``sequences`` (str or list): A single sequence or a list of sequences to predict TAD scores for.
* ``overlap_length`` (int): The number of residues shared by consecutive windows when breaking up sequences longer than 40 amino acids. Default is 39, which gives every possible window.
* ``pad`` (str): What sequences shorter than 40 amino acids are padded with. 'GS' pads with a random selection of G and S, and 'random' pads with a random selection of all 20 amino acids. Default is 'GS'.
* ``approach`` (str): Where sequences shorter than 40 amino acids are padded. 'even' pads both ends, 'N' pads only the N-terminus and 'C' pads only the C-terminus. Default is 'even'.
* ``verbose`` (bool): If True, a warning is printed when the sequences are not all 40 amino acids. Default is True.
* ``safe_mode`` (bool): If True, an exception is raised if any sequence is under 40 amino acids. If False, those sequences are padded. Default is True.
* ``seed`` (int or None): Seed for the random number generator used to choose padding residues. Set this to get the same padding (and so the same score) every time. Default is None, in which case the padding is different every time.

**Returns**:

A dictionary where each key is an input sequence and each value is a list of ``[sequence, score]`` pairs. The sequence in each pair is the exact 40 amino acid sequence the score is for, which is either a window of the input sequence or a padded version of it. The keys are in the order the sequences were passed in.

### Examples

A single sequence:

```python
predict("EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD")
{'EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD': [['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.6482064127922058]]}
```

Several sequences:

```python
predict(["QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL", "EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD"])
{'QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL': [['QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL', 0.6399572491645813]], 'EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD': [['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.648206353187561]]}
```

**Note**: The score of a sequence can change in the seventh or eighth decimal place depending on which other sequences it is predicted alongside (compare the scores for the second sequence in the two examples above). This is floating point rounding in the network and is not a meaningful difference.

A sequence longer than 40 amino acids, with windows that overlap by 20 residues:

```python
predict("EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTDEFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD", overlap_length=20)
Warning: Not all sequences are 40 amino acids long.
Sequences shorter than 40 amino acids will be padded with a random selection of G and S evenly on the N and C terminus.
Sequences longer than 40 amino acids will be windowed to make sequences 40 amino acids in length with 20 overlapping amino acids.
{'EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTDEFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD': [['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.648206353187561], ['EESFLHQSFDQSFLLSSPTDEFSPENSSSSSWSSQESFLW', 0.6744509339332581], ['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.6482064127922058]]}
```

With the default ``overlap_length`` of 39 every possible window is scored (41 windows for this 80 amino acid sequence). Smaller overlaps score fewer windows, and residues at the C-terminus are left out if the sequence does not divide evenly into windows with that overlap. The warning can be turned off with ``verbose=False``.

A sequence shorter than 40 amino acids, padded with G and S on both ends (the default) or with random residues at the N-terminus:

```python
predict("EFSPENSSSSSWSSQESFLW", safe_mode=False, verbose=False, seed=1)
{'EFSPENSSSSSWSSQESFLW': [['GGSGSSSSGGEFSPENSSSSSWSSQESFLWSGSSGSSGGS', 0.452115923166275]]}

predict("EFSPENSSSSSWSSQ", safe_mode=False, pad="random", approach="N", verbose=False, seed=1)
{'EFSPENSSSSSWSSQ': [['QPDTELILMHELRMAGRITSPECRREFSPENSSSSSWSSQ', 0.21529123187065125]]}
```

## predict_from_fasta

``predict_from_fasta`` returns TAD scores for the sequences in a FASTA file.

```python
predict_from_fasta(path_to_fasta)
```

**Parameters**:

* ``path_to_fasta`` (str): The path to the FASTA file.
* ``overlap_length``, ``pad``, ``approach``, ``verbose``, ``safe_mode`` and ``seed``: The same as for ``predict``.

**Returns**:

A dictionary where each key is the name of a sequence (its FASTA header) and each value is a two-element list: the sequence, and its list of ``[sequence, score]`` pairs (as returned by ``predict``).

```python
predict_from_fasta("tadatorch/data/testing.fasta")
{'0': ['QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL', [['QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL', 0.6399572491645813]]], '1': ['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', [['EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD', 0.648206353187561]]], '2': ['VLPPLSESFDLDSLMSTPMSSPRQNSIEAETNSSTFFDFG', [['VLPPLSESFDLDSLMSTPMSSPRQNSIEAETNSSTFFDFG', 0.6622272729873657]]], '3': ['SWLLPNSGKNSGNNNGFSIGDEFLNLVDYSSSDKQFTDQS', [['SWLLPNSGKNSGNNNGFSIGDEFLNLVDYSSSDKQFTDQS', 0.5775014162063599]]]}
```

If any sequence in the file is under 40 amino acids, ``safe_mode=False`` must be set or no predictions are made.

## predict_activation_domains

``predict_activation_domains`` annotates the activation domains in a sequence. It scores every 40 amino acid window and then calls activation domains from the scores, using one of two methods (described in detail in [How activation domains are annotated](#how-activation-domains-are-annotated)).

```python
predict_activation_domains(sequence)
```

**Parameters**:

* ``sequence`` (str): A sequence at least 40 amino acids long.
* ``threshold`` (float): Windows with a TAD score at or above this value are treated as activation domains. Default is 0.5.
* ``method`` (str): How activation domains are called from the window scores, either 'whole_window' or 'central_residues'. Default is 'whole_window'.
* ``gap_size`` (int): Gaps between activation domains shorter than this many residues are joined. Only used by the 'central_residues' method. Default is 6.
* ``min_ad_size`` (int): Activation domains shorter than this many residues are removed (after gaps have been joined). This is also the number of residues at an end of the sequence that must be marked for an activation domain to be extended to that terminus. Only used by the 'central_residues' method. Default is 6.

**Returns**:

A list with one ``ActivationDomain`` for each activation domain, in order from the N- to the C-terminus (the list is empty if there are none). An ``ActivationDomain`` has these attributes:

* ``start`` and ``end``: The positions of the first and last residues. Positions are 1-indexed and inclusive, so the domain is ``sequence[start - 1:end]``.
* ``sequence``: The sequence of the domain.
* ``number_of_windows_above_threshold``: The number of 40 amino acid windows belonging to the domain with a score at or above the threshold. With the 'whole_window' method a window belongs to a domain if it lies within it, and with the 'central_residues' method a window belongs to a domain if its two central residues do.
* ``mean_score``: The mean score of every window belonging to the domain, including any under the threshold that sit between windows above it.
* ``max_score``: The highest score of the windows belonging to the domain.

### Example

Yeast Gcn4 (281 amino acids) with each method:

```python
predict_activation_domains(gcn4_sequence)
[ActivationDomain(start=40, end=151, sequence='VGQLIFDKFIKTEEDPIIKQ...', number_of_windows_above_threshold=56, mean_score=0.5669..., max_score=0.6989...)]

predict_activation_domains(gcn4_sequence, method="central_residues")
[ActivationDomain(start=70, end=132, sequence='ALPQTATAPDAKTVLPIPELDDAVVESFFSSSTDSTPMFEYENLEDNSKEWTSLFDNDIPVTT', number_of_windows_above_threshold=54, mean_score=0.5864..., max_score=0.6989...)]
```

## Lower-level functions

These are in ``tadatorch.backend`` and are useful when you want more control than the functions above give.

* **``tadatorch.backend.predictor.predict_tada(sequences, return_both_values=False, batch_size=256)``** scores a list of sequences that are all exactly 40 amino acids long, and returns a list of scores. With ``return_both_values=True`` it returns the two outputs of the network as an array of shape (number of sequences, 2), where the first column is the TAD score and the second is 1 minus the TAD score. ``batch_size`` only affects speed and memory, not the scores.
* **``tadatorch.backend.domains.find_activation_domains(sequence, window_scores, threshold=0.5, method="whole_window", gap_size=6, min_ad_size=6)``** calls activation domains from window scores you already have, where ``window_scores[i]`` is the score of ``sequence[i:i + 40]``. Scoring is by far the slowest part, so saving the window scores and calling this function lets you try different settings without scoring again.
* **``tadatorch.backend.utils.sliding_window(sequence, window_length, overlap)``** breaks a sequence into windows, e.g. ``sliding_window(sequence, 40, 39)`` gives every 40 amino acid window.
* **``tadatorch.backend.features.create_features(sequences)``** and **``scale_features(features, load_scaler_metrics())``** calculate and scale the features the network takes as input (see [How tadatorch works](#how-tadatorch-works)).
* **``tadatorch.backend.predictor.load_model()``** returns the trained network (a PyTorch module) in evaluation mode.

For example, to score every window of a protein once and then call domains with two different settings:

```python
from tadatorch.backend.domains import find_activation_domains
from tadatorch.backend.predictor import predict_tada
from tadatorch.backend.utils import sliding_window

window_scores = predict_tada(sliding_window(gcn4_sequence, 40, 39))

find_activation_domains(gcn4_sequence, window_scores)
find_activation_domains(gcn4_sequence, window_scores, threshold=0.6, method="central_residues", gap_size=10, min_ad_size=8)
```

# Command line interface

Installing tadatorch also installs the ``tadatorch`` command, which takes a FASTA file and writes out a tab-separated (TSV) file with one row for every activation domain in the sequences in the file.

```bash
tadatorch proteins.fasta -o activation_domains.tsv
```

**Arguments**:

* ``fasta``: The path to the FASTA file.
* ``-o``, ``--output``: The path the TSV is written to. If this is not set, the TSV is written to the terminal.
* ``-t``, ``--threshold``: Windows with a TAD score at or above this value are treated as activation domains. Default is 0.5.
* ``-m``, ``--method``: How activation domains are called from the window scores, either ``whole_window`` or ``central_residues``. Default is ``whole_window``.
* ``--gap-size``: Gaps between activation domains shorter than this many residues are joined. Only used with ``--method central_residues``. Default is 6.
* ``--min-ad-size``: Activation domains shorter than this many residues are removed. This is also the number of residues at an end of a sequence that must be marked for an activation domain to be extended to that terminus. Only used with ``--method central_residues``. Default is 6.

**Columns in the TSV**:

* ``name``: The name of the sequence the activation domain is in (its full FASTA header).
* ``start`` and ``end``: The positions of the first and last residues. Positions are 1-indexed and inclusive.
* ``length``: The number of residues in the activation domain.
* ``windows_above_threshold``: The number of windows belonging to the activation domain with a score at or above the threshold.
* ``mean_score`` and ``max_score``: The mean and highest scores of the windows belonging to the activation domain (to four decimal places).
* ``sequence``: The sequence of the activation domain.

### Example

```bash
tadatorch tadatorch/data/testing.fasta -o activation_domains.tsv
[1/4] 0: 1 AD(s)
[2/4] 1: 1 AD(s)
[3/4] 2: 1 AD(s)
[4/4] 3: 1 AD(s)
Found 4 AD(s) in 4 of 4 sequence(s) using a threshold of 0.5 and the whole_window method.
```

`activation_domains.tsv` then contains:

```
name	start	end	length	windows_above_threshold	mean_score	max_score	sequence
0	1	40	40	1	0.6400	0.6400	QFNENSNIMQQQPLQGSFNPLLEYDFANHGGQWLSDYIDL
1	1	40	40	1	0.6482	0.6482	EFSPENSSSSSWSSQESFLWEESFLHQSFDQSFLLSSPTD
2	1	40	40	1	0.6622	0.6622	VLPPLSESFDLDSLMSTPMSSPRQNSIEAETNSSTFFDFG
3	1	40	40	1	0.5775	0.5775	SWLLPNSGKNSGNNNGFSIGDEFLNLVDYSSSDKQFTDQS
```

**Notes**:

* Sequences with no activation domains have no rows in the TSV.
* Sequences under 40 amino acids cannot be scored, so are skipped. The number and names of any skipped sequences are printed when the command finishes.
* Progress messages are printed separately from the TSV (to stderr), so they do not end up in the output if you redirect the TSV to a file with ``>``.
* Rows are written as each sequence finishes, so a run that is interrupted still leaves the results so far.
* The command runs on one CPU core. For very large FASTA files (for example many proteomes), it is quickest to split the file up and run several copies of the command at once.

# How activation domains are annotated

TADA scores 40 amino acid windows and does not itself say where an activation domain starts and stops, so how activation domains are called from the window scores is a choice made by tadatorch. Every window of the sequence is scored (moving one residue at a time), and the windows with a score at or above the threshold (0.5 by default) are used in one of two ways.

### whole_window (the default)

Every residue of every window at or above the threshold is part of an activation domain, and each contiguous stretch of these residues is one activation domain. Windows that overlap, or directly follow one another, therefore end up in the same activation domain.

This means activation domains are always at least 40 residues long, and an activation domain can extend up to 39 residues past the residues that are actually responsible for the score. For yeast Gcn4 this gives one activation domain at residues 40 to 151.

### central_residues

This gives tighter activation domains, in four steps:

1. For every window at or above the threshold, only its two central residues (residues 20 and 21 of the window) are marked.
2. Gaps between marked regions that are shorter than ``gap_size`` residues (6 by default) are joined.
3. Regions shorter than ``min_ad_size`` residues (6 by default) are removed. Since one window marks two residues, a region needs at least five windows in a row at or above the threshold (or several shorter runs joined across short gaps) to be kept with the default settings.
4. No window is centered on the first or last 19 residues of a sequence, so those residues cannot be marked directly. To stop activation domains at the ends of a sequence being cut short, an activation domain is extended to the N-terminus if the first ``min_ad_size`` residues that a window can mark (residues 20 to 25 by default) were all marked directly in step 1, and to the C-terminus in the same way. This means the windows at an end must consistently reach the threshold for that end to be extended: one window at the very end of a sequence is not enough, and neither are residues that are only in an activation domain because a gap was joined. Without this rule, a single window would be enough to call an activation domain of around 20 residues at a terminus, which would make activation domains much easier to call at the ends of a sequence than in the middle.

For yeast Gcn4 this gives one activation domain at residues 70 to 132, compared to 40 to 151 with ``whole_window``, and for yeast Pho4 it gives 65 to 93, compared to 46 to 112. The default ``gap_size`` and ``min_ad_size`` were chosen by intuition and have not been tuned against data.

# How tadatorch works

### Features

TADA does not see the sequence directly. Each 40 amino acid sequence is split into 36 overlapping windows of 5 residues (moving one residue at a time), and 42 features are calculated for each window, giving a 36 by 42 array per sequence (``tadatorch.backend.features.create_features``):

| Feature | What it is | Calculated with |
|---|---|---|
| 1 | Kappa (patterning of positive with respect to negative residues) of the whole 40 amino acid sequence | sparrow |
| 2 | Omega (patterning of charged and proline residues with respect to all other residues) of the whole 40 amino acid sequence | sparrow |
| 3 | Mean Kyte-Doolittle hydropathy (shifted to run from 0 to 9) | sparrow |
| 4 | Mean Wimley-White interface hydrophobicity | sparrow, with the scale TADA was trained with |
| 5 | Net charge per residue | sparrow |
| 6 | Fraction of disorder-promoting residues (T, A, G, R, D, H, Q, K, S, E, P) | sparrow |
| 7 | Fraction of charged residues | sparrow |
| 8 | Mean net charge (the absolute value of feature 5) | sparrow |
| 9 | Fraction of negative residues | sparrow |
| 10 | Fraction of positive residues | sparrow |
| 11 to 21 | The number of aliphatic (IVLA), aromatic (WFY), branching (VIT), charged (KRHDE), negative (DE), phosphorylatable (STY), polar (RKDEQNY), hydrophobic (WFLVICM), positive (KRH), sulfur containing (MC) and tiny (GASP) residues | tadatorch |
| 22 | Mean AlphaFold2 pLDDT score predicted by alphaPredict | a lookup table (see below) |
| 23 to 42 | The number of each amino acid, in the order R, K, D, E, Q, N, H, S, T, Y, C, W, M, A, I, L, F, V, P, G | tadatorch |

Kappa and Omega describe the whole 40 amino acid sequence, so they are the same in all 36 windows. Both are -1 where they are undefined (kappa when a sequence does not have both positive and negative residues, and Omega when it has no charged or proline residues). The features are then scaled to match the training data (``scale_features``), using the metrics saved when TADA was trained: each feature is standardized and then min-max scaled.

Rather than running alphaPredict for every window, the mean pLDDT score of every one of the 3.2 million possible 5-residue windows was calculated once with alphaPredict and saved (`tadatorch/data/five_mer_plddt_sums.npy`). This gives exactly the values alphaPredict would, and means alphaPredict does not need to be installed.

### The network

The network (``tadatorch.backend.model.TadaModel``) reproduces the TADA network layer for layer:

1. Two 1D convolutions (100 filters, kernel size 2), each followed by a GELU activation and dropout (dropout only applies during training). The window axis shortens from 36 to 34.
2. An attention layer, which scores each of the 34 positions, converts the scores to weights with a softmax and re-weights the positions.
3. A bidirectional LSTM (100 units in each direction) that returns its output at every position.
4. A second bidirectional LSTM (100 units in each direction) that returns only its final states.
5. A dense layer with a softmax, giving two outputs that sum to 1. The first is the TAD score.

The trained weights (430,736 values) are the original TADA weights, converted from Keras into a PyTorch state dict with `devtools/scripts/convert_keras_weights.py`. The values are unchanged; they are only rearranged to match how PyTorch lays out convolution, LSTM and dense layers.

# Accuracy and how tadatorch relates to TADA and TADA_T2

### What is the same

* **The network and its weights** (see above).
* **40 of the 42 features.** Every feature apart from kappa and Omega has the same value as in TADA_T2 (to within 0.0000000000001), and the metrics used to scale the features are the same.
* **The user-facing functions.** ``predict`` and ``predict_from_fasta`` take the same arguments and return the same format as in TADA_T2.

### What is different

* **Kappa and Omega are calculated with sparrow rather than localcider.** sparrow uses a slightly different definition of kappa that is much faster to calculate, and returns -1 (undefined) for kappa if a sequence does not have both positive and negative residues, whereas localcider (which TADA was trained with) only did so if a sequence had neither.
* **The scores are therefore close to, but not the same as, those from TADA.** For 100 sequences with scores from the original version of TADA the median difference is 0.0004 and the largest is 0.018, and for 44 sequences with scores in the TADA_T2 README the median difference is 0.005 and the largest is 0.030. Across every window of 164 yeast transcription factors (96,111 windows), 99% of scores changed by under 0.021 and 0.14% of windows moved from one side of the 0.5 threshold to the other. How well the scores predict measured activation was not affected: in the comparison with the Sanborn et al. data the ROC AUC was 0.947 with kappa and Omega from localcider and 0.948 with them from sparrow.
* **TensorFlow, localcider and alphaPredict are not needed**, and tadatorch is much faster (see [Speed](#speed)).
* ``predict_activation_domains`` and the ``tadatorch`` command are new.
* Scores are returned as Python floats rather than numpy float32 values.
* Sequences that contain anything other than the 20 standard uppercase amino acids raise an exception in ``predict``. In TADA_T2 a lowercase sequence would run but give an incorrect score, because some of the features are calculated by matching against uppercase residues.
* ``predict`` and ``predict_from_fasta`` have an extra ``seed`` argument, so the padding of sequences shorter than 40 amino acids can be made reproducible.
* The dictionary returned by ``predict`` has its keys in the order the sequences were passed in (in TADA_T2 the keys were grouped by sequence length).
* ``create_features`` only accepts sequences that are exactly 40 amino acids long.

# Speed

Scoring every 40 amino acid window of a protein takes around 0.35 milliseconds per window on one CPU core (measured on an Apple M3 Max), so a 500 amino acid protein takes under 0.2 seconds, after around a second to import tadatorch. The same calculation done as in TADA_T2 takes around 30 milliseconds per window. Around half of the remaining time is the network and around a quarter is kappa and Omega.

The speed comes from:

* calculating kappa and Omega with sparrow, for all of the windows at once;
* looking up the alphaPredict scores in a table rather than running alphaPredict;
* calculating the features of each distinct 5-residue window only once (neighbouring 40 amino acid windows share 35 of their 36 5-residue windows), and caching them;
* applying the two convolutions as matrix multiplications, which gives the same result but is around 50 times faster than PyTorch's convolution for these small inputs (which made the network as a whole three times faster);
* passing 256 sequences through the network at once.

tadatorch runs on the CPU. For large jobs, run several processes at once (for example, one per FASTA file); each process uses one core efficiently.

# Package layout

```
tadatorch/
├── tadatorch/
│   ├── __init__.py          # exposes predict, predict_from_fasta and predict_activation_domains
│   ├── tada.py              # the user-facing functions
│   ├── cli.py               # the tadatorch command
│   ├── backend/
│   │   ├── constants.py     # window sizes, the number of features and the amino acid order
│   │   ├── features.py      # calculating and scaling features
│   │   ├── model.py         # the network (TadaModel)
│   │   ├── predictor.py     # loading the network and scoring 40 amino acid sequences (predict_tada)
│   │   ├── domains.py       # calling activation domains from window scores
│   │   └── utils.py         # validating, windowing and padding sequences
│   ├── data/
│   │   ├── tada.14-0.02.pt          # the trained network weights
│   │   ├── scaler_metric.npy        # the metrics used to scale features
│   │   ├── five_mer_plddt_sums.npy  # alphaPredict scores for every 5-residue window
│   │   ├── testing.fasta            # four sequences used by the tests
│   │   └── README.md                # where each data file came from
│   └── tests/                       # the tests (pytest)
└── devtools/scripts/
    ├── convert_keras_weights.py     # converts the original Keras weights to tada.14-0.02.pt
    └── build_plddt_table.py         # builds five_mer_plddt_sums.npy with alphaPredict
```

The two scripts in `devtools/scripts/` document exactly how the data files were made, and only need to be run again if the weights or alphaPredict change.

# Development

Run the tests, the linter and the type checker from the directory holding this README:

```bash
pytest
ruff check
ruff format --check
mypy tadatorch devtools/scripts
```

The tests check tadatorch against scores from the original version of TADA and from TADA_T2 (``tadatorch/tests/reference_scores.py``), allowing for the differences that come from kappa and Omega. They also check a snapshot of tadatorch's own scores to within 0.000001 (``TADATORCH_SCORES`` in the same file), to catch any unintended change. If a change to the scores is intended (for example, if sparrow's kappa changes), the snapshot needs regenerating.

# Citation

Please cite the original TADA publication:

[Morffy, N., Van den Broeck, L., Miller, C. et al. Identification of plant transcriptional activation domains. Nature 632, 166–173 (2024)](https://doi.org/10.1038/s41586-024-07707-3)

If you use tadatorch, please also say in your methods that you used tadatorch to generate your predictions, and how activation domains were called (the method, threshold, ``gap_size`` and ``min_ad_size``), so your readers know exactly how you got your results. If you use sparrow (which tadatorch uses for kappa and Omega), please also cite it.

# License and credits

The tadatorch code is distributed under the MIT license (see LICENSE).

tadatorch is derived from [TADA_T2](https://github.com/ryanemenecker/TADA_T2) by Ryan Emenecker (Holehouse Lab, WUSM), which is also distributed under the MIT license, and TADA_T2 is a TensorFlow2 port of the original [TADA](https://github.com/LisaVdB/TADA) by Lisa Van den Broeck. The trained network weights (`tada.14-0.02.pt`, converted from the original Keras weights) and the feature scaling metrics (`scaler_metric.npy`) come from TADA by way of TADA_T2. The original TADA repository does not include a license file.

TADA_T2 (and so the layout of this package) was based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.

# Version history

### October 2026

* First version of tadatorch.
