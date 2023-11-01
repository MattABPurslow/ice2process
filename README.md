# ice2process
Scripts used to process observed and simulated ICESat-2 data for use in [Purslow et al (2023)](https://doi.org/10.1016/j.srs.2023.100086).

## Key Scripts
### atl03.py
Load ATL03 and ATL08 data into dataframes.

### compareAll.py
Compare simuated and observed ICESat-2 Plant Structural Traits at all sites.

### compareALS.py
Assess the impact of ρv/ρg on ICESat-2 canopy cover bias with respect to ALS canopy cover.

### footprints.py
Extract ICESat-2 footprint locations.

### geoid.py
Apply geoid correction to ALS data.

### metrics.py
Calculate Plant Structural Traits for simulated and observed ICESat-2 data.

### offset.py
Correct for geolocation offset between ICESat-2 and ALS data.

### params.py
Define file locations and site-specific parameters.

### photons.py
Reclassify ICESat-2 data using ALS classification.

### poisson.py
Assess the photon distribution of observed ICESat-2 data.

### pseudowaveDiagram.py
Produce diagram showing the generation of a pseudowavefrom from an ICESat-2 segment.

### pulse.py
Extract ICESat-2 pulse shape from ATL03.

### rates.py
Calculate ICESat-2 photon rates and ρv/ρg.

### segment.py
Collect simulated ICESat-2 data into ATL08 segments.

### simulate.py
Wrapper for the ICESat-2 simulator (https://bitbucket.org/StevenHancock/gedisimulator/).

### simulatorDiagram.py
Create diagram to demonstrate the simulator workflow.

### siteStats.py
Extract mean ALS statistics for each site.

### tiles.py
Identify ALS tiles intersected by ICESat-2 tracks.

### tracks.py
Identify ICESat-2 tracks intersecting with ALS boundary.

### waveDiagram.py
Produce diagram showing relationship between full waveform and photon counting data.
waves.py
