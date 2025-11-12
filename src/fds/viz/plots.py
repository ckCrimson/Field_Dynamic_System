from __future__ import annotations
from typing import Optional
# keep dependencies optional; import only when used

def plot_field_values(field, *, show: bool = True, ax=None):
    """
    Minimal placeholder: later, map states->coords and values->colormap.
    Defer matplotlib import to runtime so fds installs without viz deps.
    """
    import matplotlib.pyplot as plt  # lazy import
    ax = ax or plt.gca()
    # TODO: implement once Field exposes as_arrays() (states, values)
    # ax.scatter(x, y, c=val, s=...)
    if show:
        plt.show()
    return ax
