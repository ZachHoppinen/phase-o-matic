# phase-o-matic
Python package for calculating Interferometric Synthetic Aperture Radar phase delays from ERA5 atmospheric models. Utilizes x-array to easily download, processes, and add phase delays to netcdfs of InSAR Phase. 

Useful publications for this repo:
 - Doin, M.-P., Lasserre, C., Peltzer, G., Cavalié, O. & Doubre, C. Corrections of stratified tropospheric delays in SAR interferometry: Validation with global atmospheric models. J Appl Geophys 69, 35–50 (2009).

 - Jolivet, R. et al. Improving InSAR geodesy using Global Atmospheric Models. J Geophys Res Solid Earth 119, 2324–2341 (2014).

 - Hu, Z. & Mallorquí, J. J. An Accurate Method to Correct Atmospheric Phase Delay for InSAR with the ERA5 Global Atmospheric Model. Remote Sens-basel 11, 1969 (2019).

This code is adapted from the awesome atmospheric phase delay repo: https://github.com/insarlab/PyAPS utilizing xarray.

<img src="https://github.com/ZachKeskinen/phase-o-matic/blob/main/images/pyaps_phaseo_compare.png">
