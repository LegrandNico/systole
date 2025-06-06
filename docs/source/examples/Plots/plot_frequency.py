"""
Plot frequency
==============

"""


# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Licence: GPL v3

# %%
# Visualizing HRV frequency domain from RR time series using Matplotlib as plotting backend
# -----------------------------------------------------------------------------------------
from systole import import_rr
from systole.plots import plot_frequency

# Import PPG recording as numpy array
rr = import_rr().rr.to_numpy()
plot_frequency(rr, input_type="rr_ms")

# %%
# Visualizing HRV frequency domain from RR time series using Bokeh as plotting backend
# ------------------------------------------------------------------------------------
from systole import import_rr
from systole.plots import plot_frequency
from bokeh.io import output_notebook
from bokeh.plotting import show

output_notebook()

show(plot_frequency(rr, input_type="rr_ms", backend="bokeh"))
