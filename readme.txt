Contains code for figures and statistics for the publication:

Márton Albert Hajnal, Zsombor Szabó, Andrea Albert, Duy Tran, Karen Safaryan, Michael Einstein, Mauricio Vallejo Martelo, Pierre-Olivier Polack, Peyman Golshani, Gergő Orbán
"Shifts in attention drive context-dependent subspace encoding in anterior cingulate cortex during decision making"

The code runs on julia (tested on versions 1.8-1.9).
Project.toml and Manifest.toml files are provided in the root folder of this repository.
All folder references below are relative to the sourc efolders.



The code is to be run by > julia -t auto --project=. src/run.jl [OPTIONS]

Configuration files:
params.yaml and params-rnn.yaml: setup file for modules, sampling constants,
experiment trial structure, output files, and the rnn model parameters.


The code is in src/
options.jl lists possible options. If no -s mouse given, all mice are used in all calculations
run.py: contains the main branching caller routine, based on command line options
preprocess.jl: handles mouse experiment specific loaders, generators
figure.jl: will produce figure files from precalculated and cached results data

Caches can be recalculated using the following files containing referenced calculation
routines in thematic groups. The figure.jl file should guide the reader as to
which calculation routines are needed by searching for the cached results files
within cache/ as .bson binary files in the calculation files.
physiology.jl
subspaces.jl
nrnmodels.jl

There are additional common neuroscience routines grouped in the module NeuroscienceCommon.jl and common/
The nwb wrapper needs PyCall.jl; the easiest way to use the julia in-built python environment.
It needs to be built with ENV["PYTHON"]=""

The following folder structure is necessary to be created before being able to cache intermediate
calculations, results and generate output:
cache/*
results/*
figures/
where * should be inferred from the command line options in options.jl for the scripts called from run.jl.


We provide spike sorted and behavioural data in nwb format (.nwb files) for each mouse.
https://www.nwb.org/
We also provide pretrained RNN models in .bson binary files.
This two type of data can be downloaded from a single Zenodo data repository:
doi://10.5281/zenodo.8379272
https://zenodo.org/record/8379272
Mouse data files should be downloaded into a data/mouse/nwb/ folder.
RNN zip file should be unzipped into a data/model/rnn/ folder retaining
folder structure in the zip file.



Questions about the code and data should be addressed to: hajnal.marton@wigner.mta.hu
