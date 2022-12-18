import zipfile
import soundfile as sf
import torch
import kenlm
import IPython
import librosa

from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from pydub import AudioSegment
import math

import glob, os
from underthesea import text_normalize
from underthesea import word_tokenize
from syllable import *

processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

cache_dir = './temp/'
#please uncomment this and run this, second time -> : uncomment this
#=========================================================================================================
# lang_model = hf_hub_download("nguyenvulebinh/wav2vec2-base-vietnamese-250h", filename='vi_lm_4grams.bin.zip')

# with zipfile.ZipFile(lang_model, 'r') as z:
#     z.extractall(cache_dir)
#=========================================================================================================

lang_model = cache_dir + 'vi_lm_4grams.bin'
def fetch_lang_model_decoder(tokenizer, ngram_lm_path):
    dictionary = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in dictionary.items())

    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab

    vocab_list[tokenizer.pad_token_id] = ""
    vocab_list[tokenizer.unk_token_id] = ""

    vocab_list[tokenizer.word_delimiter_token_id] = " "
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)

    lm_model = kenlm.Model(ngram_lm_path)

    decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
    return decoder

ngram_lm_model = fetch_lang_model_decoder(processor.tokenizer, lang_model)

class WavSplitter():
    def __init__(self, obj):
        self.obj = obj
        self.audio = AudioSegment.from_wav(self.obj)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export('./' + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        for i in range(0, math.ceil(self.get_duration()), sec_per_split):
            split_fn = str(i) + '_sample.wav'
            self.single_split(i, i + sec_per_split, split_fn)
            
            print('Segment ' + str(i) + ' ...done.')
            
            if i ==  math.ceil(self.get_duration()) - sec_per_split:
                print('All splited successfully')

def map_to_array(sound):
    speech, sampling_rate = librosa.load(sound["file"],sr=16000)
    sound["speech"] = speech
    sound["sampling_rate"] = sampling_rate
    return sound

def collect_sample():
    os.chdir(".")
    audios = [file for file in glob.glob("*.wav") if "_sample" in file]
    audios.sort(key= lambda i: int(i.rstrip('_sample.wav')))

    return audios
  
def remove_temp_file():
    for file in glob.glob("*.wav"):
        print(file)
        if "_sample" in file:
           os.remove(file)

def get_output_raw(audios):
    raw = []
    for audio in audios:
        data = map_to_array({"file": audio})

        input_values = processor(
            data["speech"], 
            sampling_rate=data["sampling_rate"], 
            return_tensors="pt"
        ).input_values

        logits = model(input_values).logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        
        raw.append(processor.decode(pred_ids))
    return raw
    
def get_output_lm(audios):
    lm = []
    for audio in audios:
        data = map_to_array({"file": audio})

        input_values = processor(
            data["speech"], 
            sampling_rate=data["sampling_rate"], 
            return_tensors="pt"
        ).input_values

        logits = model(input_values).logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        
        lm.append(ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500))
    return lm
    
 