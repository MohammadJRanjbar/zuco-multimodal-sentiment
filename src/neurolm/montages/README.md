# Electrode coordinate sources

| file | source | notes |
| --- | --- | --- |
| `zuco_chanlocs.csv` | `EEG.chanlocs` in `matlab/gip_ZAB_SR5_EEG.mat` of the ZuCo authors' repository [norahollenstein/zuco-benchmark](https://github.com/norahollenstein/zuco-benchmark) (SHA-256 `c42eaa3d234f14d667a9fbded19a5d5cdcb78375e40ca093a471d3afc5a84960`) | The authors load this file to plot `results*_SR.mat` channel vectors, so its 105-row order is the `rawData` channel order. `labels` and coordinates are copied verbatim; `zab_sr5_std` is the per-channel std of that file's `EEG.data` (Cz is exactly 0). `EEG.chaninfo.filename` is `GSN-HydroCel-129.sfp`; chanlocs X = sfp Y and chanlocs Y = −sfp X. |
| `zuco_fiducials.csv` | `EEG.chaninfo.nodatchans` of the same file | FidNz, FidT9, FidT10. |
| `GSN-HydroCel-129.sfp` | MNE-Python v1.8.0 `mne/channels/data/montages/` (BSD-3-Clause) | EGI HydroCel template, identical to the file the ZuCo chanlocs were built from. |
| `standard_1005.elc` | MNE-Python v1.8.0 `mne/channels/data/montages/` (BSD-3-Clause) | 10-05 template positions (Oostenveld & Praamstra, 2001). |

ZuCo keeps 104 of E1–E128 plus the vertex reference Cz. The 24 electrodes not
in `rawData` are E1, E8, E14, E17, E21, E25, E32, E48, E49, E56, E63, E68, E73,
E81, E88, E94, E99, E107, E113, E119, E125, E126, E127, E128 (face, neck, and
EOG sites).
