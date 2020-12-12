# Denoising EEG Signals using ICA

Justin Nguyen, Mingxiao An, Siddharth Mehta, Yi-Chia Wu

Semester Project for 18-797 with Dr. Bhiksha Raj

## Project Structure

Our project is centered around the `BuildingBlocks` directory which contains
our single subject experiments and "building blocks" for putting together
MI denoising and classification trials across the entire dataset.

The directories: `BaselineClassification`, `BlinkingDenoiseClassification`, 
`EyeMovementLRDenoiseClassification`, `EyeMovementUDDenoiseClassification`,
`HeadMovementDenoiseClassification`, `JawMovementDenoiseClassification` contain
our denoising and classification trails for each noise type.

`Cho2017.py` defines the class for processing the dataset and `plotting.ipynb`
is a notebook which generates the plots for our presentation and reports.

## Contributions

Justin Nguyen: `Cho2017.py`, EEG experiments in `BuildingBlocks`, and the starter
notebooks for the Corrmap denoising notebooks for the entire dataset trials.

Mingxiao An: `plotting.ipynb` and all the bug fixes relating to getting the ICA
de-noised signal and the classifier (CSP+LDA) working.

Siddarth Mehta: MI-Classifier notebooks and setting up the classification pipeline
and setting up the dataset hold-out, crossvalidation, and testing.

Yi-Chia Wu: MI-Classifier notebooks and transferred starter notebooks for use on
just blink artifacts to eye-movement, head-movement, and jaw-movement noise types.
