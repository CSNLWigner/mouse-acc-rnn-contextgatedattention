Contains code for figures and statistics for the publication:

Márton Albert Hajnal, Zsombor Szabó, Andrea Albert, Duy Tran, Karen Safaryan, Michael Einstein, Mauricio Vallejo Martelo, Pierre-Olivier Polack, Peyman Golshani, Gergő Orbán
"Shifts in attention drive context-dependent subspace encoding in anterior cingulate cortex during decision making"



figure.jl: will produce figure files from precalculated and cached results data

caches can be recalculated using the following files:

params.yaml and params-rnn.yaml: setup file for modules, sampling constants, experiment trial structure, output files, and the rnn model parameters.
run.py: command center, choose which mice to include in a given analysis and choose which calculation routines to run

preprocess.jl: handles mouse experiment specific loaders for raw and cached data

All other files contain above referenced calculation routines in thematic groups

The figure.jl file should guide the reader as to which calculation routines are needed by searching
for the cached results files within ../cache as .bson binary files in the calculation files



The following folder structure is necessary to be created before being able to cache calculation results and generate output:
cache/*
results/*
where * should be inferred from the command line options for the scripts in options.jl and run.jl.


We provide spike sorted and behavioural data in nwb format (.nwb files) for each mouse.
https://www.nwb.org/
We also provide pretrained RNN models in .bson binary files.
Data can be downloaded from the Zenodo data repository:
doi://10.5281/zenodo.8379272
https://zenodo.org/record/8379272
Mouse data should be downloaded into data/mouse/nwb/ folder.
RNN zip file should be unzipped into data/model/rnn/ folder.



Questions about the code and data should be addressed to: hajnal.marton@wigner.mta.hu
