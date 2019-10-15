# RGAN

Use generative adversarial networks (GANs) to generate real-valued time series. The GAN uses RNNs for both encoder and decoder (specifically LSTMs). 

## Example: sine waves

Primary dependencies: `tensorflow`, `scipy`, `numpy`, `pandas`. See `requirements.txt` for specific versions. 

This code is tested on both Python 2.7.16 and Python 3.6.6.

Simplest route to running code (Linux/Mac):
```
git clone git@github.com:ratschlab/RGAN.git
cd RGAN
mkdir experiments/parameters experiments/data
python experiment.py --settings_file sine
```
Here, `experiments/settings/sine.txt` is the settings file for generating sine waves. 

Evolution of Discriminator (D) and Generator (G ) training loss:

![experiments](experiments/traces/sine_trace.png)

Random samples of synthetic sine waves generated at each epoch:

![experiments](experiments/plots/sine-animation.gif)

Random sample of real sine waves:

![experiments](experiments/plots/sine_real_epoch0000.png)

Frequency and amplitude distributions of real and generated sine waves:

![experiments](experiments/plots/sine_eval0000.png)

.csv files of real and generated sine waves saved to `experiments/data`.

## Example: MNIST as a time-series

Get MNIST as CSVs here: https://pjreddie.com/projects/mnist-in-csv/

```
python experiment.py --settings_file mnistfull
```
<!-- Evolution of Discriminator (D) and Generator (G ) training loss:

![experiments](experiments/traces/mnistfull_trace.png)

-->
Random samples of synthetic MNIST digits generated at each epoch:

![experiments](experiments/plots/mnist-animation.gif) 
<!-- 
Random sample of real MNIST digits:

![experiments](experiments/plots/mnistfull_real_epoch0000.png) -->

## Files in this Repository

The main script is `experiment.py` - this parses many options, loads and preprocesses data as needed, trains a model, and does evaluation. It does this by calling on some helper scripts:
- `data_utils.py`: utilities pertaining to data: generating toy data (e.g. sine waves, GP samples), loading MNIST and eICU data, doing test/train split, normalising data, generating synthetic data to use in TSTR experiments
- `model.py`: functions for defining ML models, i.e. the tensorflow meat, defines the generator and discriminator, the update steps, and functions for sampling from the model and 'inverting' points to find their latent-space representations
- `plotting.py`: visualisation scripts using matplotlib
- `mmd.py`: for maximum-mean discrepancy calculations, mostly taken from https://github.com/dougalsutherland/opt-mmd

Other scripts in the repo:
- `eval.py`: functions for evaluating the RGAN/generated data, like testing if the RGAN has memorised the training data, comparing two models, getting reconstruction errors, and generating data for visualistions of things like varying the latent dimensions, interpolating between input samples 
- `mod_core_rnn_cell_impl.py`: this is a modification of the same script from TensorFlow, modified to allow us to initialise the bias in the LSTM (required for saving/loading models)
- `kernel.py`: some playing around with kernels on time series
- `tf_ops.py`: required by `eugenium_mmd.py`

## Acknowledgements

This repository is forked from [ratschlab/RGAN](github.com/ratschlab/RGAN) the repo for the paper, [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)_, by Stephanie L. Hyland* ([@corcra](https://github.com/corcra)), Cristóbal Esteban* ([@cresteban](https://github.com/cresteban)), and Gunnar Rätsch ([@ratsch](https://github.com/ratsch)), from the Ratschlab, also known as the [Biomedical Informatics](http://bmi.inf.ethz.ch/) Group at ETH Zurich.