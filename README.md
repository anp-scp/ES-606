## Code for the project submitted for partial requirement for the course ES 606

* * * 

### Notebooks for pre-processing

* Check [notebooks/etlBandPass.ipynb](notebooks/etlBandPass.ipynb) and [notebooks/etlBandPassAllChannels.ipynb](notebooks/etlBandPassAllChannels.ipynb) for loading dataset and applying band pass filter.
* Check [notebooks/etlBandPassCWT.ipynb](notebooks/etlBandPassCWT.ipynb) for performing Continuous Wavelet Transform

Note: The above files are listed as per the order of dependency

* * *

### Notebooks for training models

* Check [notebooks/dlPipelineCNN001.ipynb](notebooks/dlPipelineCNN001.ipynb) for EEG classification with 4 channels using architecture: CNN001
* Check [notebooks/dlPipelineCNN002_4channels_scalogram.ipynb](notebooks/dlPipelineCNN002_4channels_scalogram.ipynb) for EEG classification with scalograms of EEG recording from 4 channels using architecture: CNN002
* Check [notebooks/dlPipelineCNN002_14channels_scalogram.ipynb](notebooks/dlPipelineCNN002_14channels_scalogram.ipynb) for EEG classification with scalograms of EEG recordings from 14 channels using architecture: CNN002
* Check [notebooks/dlPipelineVGG_4channels_scalogram.ipynb](notebooks/dlPipelineVGG_4channels_scalogram.ipynb) for EEG classification with scalograms of EEG recording from 4 channels using architecture: VGG_Style_3Block
* Check [notebooks/dlPipelineVGG_14channels_scalogram.ipynb](notebooks/dlPipelineVGG_14channels_scalogram.ipynb) for EEG classification with scalograms of EEG recordings from 14 channels using architecture: VGG_Style_3Block

* * *

Dataset: [MNIST of Brain digits](http://mindbigdata.com/opendb/)

* * *

### View plots for all experiments in below links

* Plots for EEG classification with 4 channels using architecture CNN001: [Click Here](https://tensorboard.dev/experiment/ZWjZtbPwSlWmAKE7t8C9kg/)
* Plots for EEG classification with scalograms of EEG recording from 4 channels using architecture CNN002: [Click Here](https://tensorboard.dev/experiment/bRMCxEn4S92du8bkMa5ZRg/)
* Plots for EEG classification with scalograms of EEG recordings from 14 channels using architecture CNN002: [Click Here](https://tensorboard.dev/experiment/0VVXbIqPS4qf7KygesN2pQ/)
* Plots for EEG classification with scalograms of EEG recording from 4 channels using architecture VGG_Style_3Block [Click Here](https://tensorboard.dev/experiment/pD3UMzMiQJKmtGphvOLkzg/)
* Plots for EEG classification with scalograms of EEG recordings from 14 channels using architecture VGG_Style_3Block [Click Here](https://tensorboard.dev/experiment/med1xsbkRVWiN9VMvI3V5A/)