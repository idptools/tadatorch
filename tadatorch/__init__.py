"""A PyTorch re-implementation of the TADA transcriptional activation domain predictor."""

from importlib.metadata import PackageNotFoundError, version

from tadatorch.tada import predict, predict_activation_domains, predict_from_fasta

__all__ = ["predict", "predict_activation_domains", "predict_from_fasta"]

try:
    __version__ = version("tadatorch")
except PackageNotFoundError:
    # tadatorch is being run from a source directory without being installed,
    # so there is no installed version to report
    __version__ = "unknown"
