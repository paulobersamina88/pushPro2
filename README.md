# MDOF RSA–Pushover Reconciliation PRO

This version keeps the MDOF concept and manual checking:

- Input floor weight and storey stiffness from STAAD.
- Compute modal properties and RSA base shear.
- Use first-mode pushover force pattern.
- Input beam/column plastic moment per frame.
- Indicate number of frames in the analyzed axis.
- Compute storey yield capacity from column and beam mechanisms.
- Generate nonlinear MDOF pushover curve using storey spring backbone.
- Convert pushover to ADRS and compare to response spectrum demand.

## Deploy online

Upload `app.py` and `requirements.txt` to GitHub, then deploy in Streamlit Community Cloud.
Main file path: `app.py`.
