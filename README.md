# A TensorFlow Implementation of DC-TTS:

Implementation of the DCTTS introduced in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy
  * keras (Not yet, i'm still writing a Keras version)

## Data

I train English models on the LJ Speech dataset 
* [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available, and it has 24 hours of reasonable quality samples.
Nick's and Kate's audiobooks are additionally used to see if the model can learn even with less data, variable speech samples. They are 18 hours and 5 hours long, respectively. Finally, KSS Dataset is a Korean single speaker speech dataset that lasts more than 12 hours.


## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. (If you want to do preprocessing, set prepro True`.
  * STEP 2. Run `python train.py 1` for training Text2Mel. (If you set prepro True, run python prepro.py first)
  * STEP 3. Run `python train.py 2` for training SSRN.

You can do STEP 2 and 3 at the same time, if you have more than one gpu card.

## Training Curves

<img src="fig/training_curves.png">

## Attention Plot
<img src="fig/attention.gif">

## Sample Synthesis
I generate speech samples based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html) as the original paper does. It is already included in the repo.

  * Run `synthesize.py` and check the files in `samples`.
  
## Notes

  * The original code is from [Kyubyong](https://github.com/Kyubyong/dc_tts), i just modified some things for my needs