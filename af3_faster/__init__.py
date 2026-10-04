"""af3-faster: in-process fast mode for an existing AlphaFold 3 install.

Works on alphafold3-colabfold (OF3 / openbind0 included) and on google-deepmind/alphafold3
(AF3 weights only). Does not edit the native tree. `python -m af3_faster` / `af3-faster`
installs the pair kernels and Pallas levers, then runs native `run_alphafold.py`.
"""
__version__ = "0.2.0"
PREFIX = "[af3-faster]"
EXIT_NOT_ACTIVE = 3
FAST_MODELS = ("alphafold3", "openfold3", "openbind0")
OF3_MODELS = ("openfold3", "openbind0")
