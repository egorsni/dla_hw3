# FastSpeech2


https://github.com/egorsni/dla_hw3/assets/43299958/3266ad51-e795-4eb0-9da0-f173e429be8b


## Installation guide

Run bash install from seminar notebook

```
#install libraries
pip install torchaudio
pip install wandb
pip install gdown

#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

#download Waveglow
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
echo $(ls mels | wc -l)

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null

# we will use waveglow code, data and audio preprocessing from this repo
git clone https://github.com/xcmyz/FastSpeech.git
mv FastSpeech/text .
mv FastSpeech/audio .
mv FastSpeech/waveglow/* waveglow/
mv FastSpeech/utils.py .
mv FastSpeech/glow.py .
```

install requirenments

```
pip install -r requirements.txt
```

load model

```
python3 load_checkpoints.py
```

run test

```
python3 test.py -c hw_asr/configs/one_batch_test_resume.json -r saved/models/one_batch_test/checkpoints/model_best/best_model.pth -t ./test_texts.txt
```

you can write your own texts in ```./test_texts.txt``` or  use ```-t {path_to_your_texts}```. Your texts will be splitted by \n

results will be in folder ```./test_results/{alpha}/{beta}/{gamma}/s={wav_index}_waveglow_alpha_{alpha}_beta_{beta}_gamma_{gamma}.wav```
