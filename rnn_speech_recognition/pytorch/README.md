# 1. Problem 
Speech recognition accepts raw audio samples and produces a corresponding text transcription.

# 2. Directions

## Steps to configure machine
### From Docker

```
git clone https://github.com/mlcommon/training.git
```
2. Install CUDA and Docker
```
source training/install_cuda_docker.sh
```
3. Build the docker image for the single stage detection task
```
# Build from Dockerfile
cd training/rnn_speech_recognition/pytorch/
bash scripts/docker/build.sh
```

#### Requirements
Currently, the reference uses CUDA-11.0 (see [Dockerfile](Dockerfile#L15)).
Here you can find a table listing compatible drivers: https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver

## Steps to download data
1. Start an interactive session in the NGC container to run data download/training/inference
```
bash scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULTS_DIR>
```

Within the container, the contents of this repository will be copied to the `/workspace/rnnt` directory. The `/datasets`, `/checkpoints`, `/results` directories are mounted as volumes
and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<RESULT_DIR>` on the host.

2. Download and preprocess the dataset.

No GPU is required for data download and preprocessing. Therefore, if GPU usage is a limited resource, launch the container for this section on a CPU machine by following prevoius steps.

Note: Downloading and preprocessing the dataset requires 500GB of free disk space and can take several hours to complete.

This repository provides scripts to download, and extract the following datasets:

* LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)

LibriSpeech contains 1000 hours of 16kHz read English speech derived from public domain audiobooks from LibriVox project and has been carefully segmented and aligned. For more information, see the [LIBRISPEECH: AN ASR CORPUS BASED ON PUBLIC DOMAIN AUDIO BOOKS](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) paper.

Inside the container, download and extract the datasets into the required format for later training and inference:
```bash
bash scripts/download_librispeech.sh
```
Once the data download is complete, the following folders should exist:

* `/datasets/LibriSpeech/`
   * `train-clean-100/`
   * `train-clean-360/`
   * `train-other-500/`
   * `dev-clean/`
   * `dev-other/`
   * `test-clean/`
   * `test-other/`

Since `/datasets/` is mounted to `<DATA_DIR>` on the host (see Step 3),  once the dataset is downloaded it will be accessible from outside of the container at `<DATA_DIR>/LibriSpeech`.

Next, convert the data into WAV files:
```bash
bash scripts/preprocess_librispeech.sh
```
Once the data is converted, the following additional files and folders should exist:
* `datasets/LibriSpeech/`
   * `librispeech-train-clean-100-wav.json`
   * `librispeech-train-clean-360-wav.json`
   * `librispeech-train-other-500-wav.json`
   * `librispeech-dev-clean-wav.json`
   * `librispeech-dev-other-wav.json`
   * `librispeech-test-clean-wav.json`
   * `librispeech-test-other-wav.json`
   * `train-clean-100-wav/`
   * `train-clean-360-wav/`
   * `train-other-500-wav/`
   * `dev-clean-wav/`
   * `dev-other-wav/`
   * `test-clean-wav/`
   * `test-other-wav/`

## Steps to run benchmark.

### Steps to launch training

Inside the container, use the following script to start training.
Make sure the downloaded and preprocessed dataset is located at `<DATA_DIR>/LibriSpeech` on the host (see Step 3), which corresponds to `/datasets/LibriSpeech` inside the container.

```bash
bash scripts/train.sh
```

This script tries to use 8 GPUs by default.
To run 1-gpu training, use the following command:

```bash
NUM_GPUS=1 GRADIENT_ACCUMULATION_STEPS=64 scripts/train.sh
```

# 3. Dataset/Environment
### Publication/Attribution
["OpenSLR LibriSpeech Corpus"](http://www.openslr.org/12/) provides over 1000 hours of speech data in the form of raw audio.
### Data preprocessing
Data preprocessing is described by scripts mentioned in the [Steps to download data](#steps-to-download-data).
### Data pipeline
Transcripts are encoded to sentencepieces using model produced in [Steps to download data](#steps-to-download-data).
Audio processing consists of the following steps:
1. audio is decoded with sample rate choosen uniformly between 13800 and 18400;
2. silience is trimmed with -60 dB threshold;
3. random noise with normal distribution and 0.00001 amplitude is applied;
4. Pre-emphasis filter is applied;
1. spectograms are calculated with 512 ffts, 20ms window and 10ms stride;
1. MelFilterBanks are calculated with 80 features and normalization;
1. features are translated to decibeles with log(10) multiplier reference magnitude 1 and 1e-20 cutoff;
1. 
### Training and test data separation
Dataset authors separated it to test and training subsets. For this benchmark, training is done on train-clean-100, train-clean-360 and train-other-500 subsets. Evaluation is done on dev-clean subset.
### Training data order
To reduce data padding in minibatches, data bucketing is applied.
The algorithm is implemented here:
[link](https://github.com/mlcommons/training/blob/2126999a1ffff542064bb3208650a1e673920dcf/rnn_speech_recognition/pytorch/common/data/dali/sampler.py#L65-L105)
and can be described as follows:
0. drop samples than given threshold;
1. sort data by audio length;
2. split data into 6 equally sized buckets;
3. for every epochs:
    1. shuffle data in each bucket;
    2. as long as all samples are not divisible by global batch size, remove random element from random bucket;
    3. concatenate all buckets;
    4. split samples into minibatches;
    5. shuffle minibatches in the epoch.

### Test data order
Test data order is the same as in the dataset.
# 4. Model
### Publication/Attribution
Cite paper describing model plus any additional attribution requested by code authors 
### List of layers 
Brief summary of structure of model
### Weight and bias initialization
How are weights and biases initialized
### Loss function
Transducer Loss
### Optimizer
TBD, currently Adam
# 5. Quality
### Quality metric
Word Error Rate (WER) across all words in the output text of all samples in the validation set.
### Quality target
What is the numeric quality target
### Evaluation frequency
TBD
### Evaluation thoroughness
TBD
