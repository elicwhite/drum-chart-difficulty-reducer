"""
Vendored, standalone baseline drum-difficulty reducers -- independent
reimplementations of the HOPCAT and Onyx tools, adapted to run directly on
this repo's own ms-based parsed-chart dict (no MIDI files, no research-tree
dependency). See hopcat.py / onyx.py for the conversion approach and
documented divergences from the real tools' MIDI-based behavior, and
_hopcat_algo.py / _onyx_algo.py for the ported algorithms themselves.
"""

from .hopcat import reduce_hopcat
from .onyx import reduce_onyx

__all__ = ["reduce_hopcat", "reduce_onyx"]
