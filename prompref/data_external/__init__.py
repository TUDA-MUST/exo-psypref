"""Hip moments from public gait datasets, as Trials the population generator can use.

The simulated users and the prior both come from Molinaro's five subjects. This package
converts walking data recorded in other labs and populations into the same
``preftune.data.loader.Trial`` objects, so that users can be generated from them while the
prior stays the Molinaro one:

* :mod:`.fukuchi` -- Fukuchi, Fukuchi & Duarte (2018), PeerJ 6:e4640: young and older
  adults, treadmill (8 speeds) and overground (3 speeds);
* :mod:`.reznick` -- Reznick et al. (2021), Sci. Data 8:282: ten young adults on a
  treadmill at 3 speeds and 5 inclines.

Raw data never enter the repository. They live under ``EXTERNAL_GAIT_DATA`` (default
``.../02__DataAnalysis/external_gait_data``), one folder per dataset.
"""

from .common import ExternalTrial, data_root, mean_cycle, trial_from_mean_cycle, trial_from_strides

__all__ = ["ExternalTrial", "data_root", "mean_cycle", "trial_from_mean_cycle",
           "trial_from_strides"]
