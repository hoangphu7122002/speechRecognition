import io
import librosa
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import pydub
import streamlit_ext as ste
import gridfs
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from model import *
import time

FILENAME = None
user_id = "639f0f448ad4e552ffb17d43"
cluster = MongoClient('mongodb+srv://hoangphu7122002:071202@cluster0.obabt.mongodb.net/?retryWrites=true&w=majority')
db = cluster["audio_demo"]
user_find_id = ObjectId(user_id)

plt.rcParams["figure.figsize"] = (10, 7)

tones = {
   "huyền" : "`",
   "sắc": "´",
   "ngã" : "~",
   "nặng" : ".",
   "hỏi" : "?",
   "ngang" : "_" 
}

def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


@st.cache
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()

def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()

def processing(file_uploader):
    #split dataset
    wav_splitter = WavSplitter(file_uploader)
    wav_splitter.multiple_split(sec_per_split=8)
    
    audios = collect_sample()
    
    # raw = get_output_raw(audios)
    lm = get_output_lm(audios)
    remove_temp_file()
    # [print("R: {}".format(r),  sep='\n') for r in raw]
    # print("\n")
    # [print("P: {}".format(wl),  sep='\n') for wl in lm]

    return lm
def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)

def level_sen(lm):
    sentence = ""
    for wl in lm:
        sentence = sentence + " " + wl
    
    sentence = sentence.strip()
    print(sentence)
    return sentence
    
def level_syllable(sentence):
    print("===================")
    print(sentence)
    print("===================")
    words = sentence.split(' ')
    sents = ""
    
    for word in words:
        #print(split_vietnamese_syllable(word))
        try:
            first, second, tone = split_vietnamese_syllable(word)
            if first == '':
              #   print(tones[tone])
               sents = sents + ' {}/{}'.format(second,tones[tone])
            else:
               sents = sents + ' {}/{}/{}'.format(first,second,tones[tone])
        except:
            continue
    sents = sents.strip()
    return sents

def level_word(sentence):
    sentence = text_normalize(sentence)
    words_token = word_tokenize(sentence, format="text")
    
    return words_token
    
def plot_audio_transformations(y, sr, pipeline, file_uploader = None):
    cols = [1, 1, 1]
    if file_uploader != None:
        print("===============================\n")
        print(file_uploader,type(file_uploader))
        print("===============================\n")
    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")
    y = y
    sr = sr
    lm = processing(file_uploader)
    sentence = level_sen(lm)
    
    global FILENAME
    for choose in pipeline:
        if choose == 0:
            st.markdown(
                f"<h4 style='text-align: left; color: black;'>Syllable</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            syllable = level_syllable(sentence)
            st.text("content")
            st.write(syllable)
            check = ste.download_button('Download syllable text', syllable)  
            # check = st.button('save syllable')
            if check:
                fs = gridfs.GridFS(db)
                fs.put(file_uploader.getvalue(), filename=FILENAME)
                audio_file = db["fs.files"]
                update = {"$set": {"user_id": user_find_id}}
                audio_file.update_one({}, update)
                audio = audio_file.find_one({})
                audio_id = ObjectId(audio["_id"])
                audio_label = db["audio_label"]
                audio_label.insert_one({"audio_id": audio_id, 
                        "level": 3, 
                        "content": syllable,
                        "approve": 1})
        elif choose == 1:
            st.markdown(
                f"<h4 style='text-align: left; color: black;'>Word</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            word = level_word(sentence)
            st.text("content")
            st.write(word)
            check = ste.download_button('Download word text', word) 
            if check:
                fs = gridfs.GridFS(db)
                
                fs.put(file_uploader.getvalue(), filename=FILENAME)
                audio_file = db["fs.files"]
                update = {"$set": {"user_id": user_find_id}}
                audio_file.update_one({}, update)
                audio = audio_file.find_one({})
                audio_id = ObjectId(audio["_id"])
                audio_label = db["audio_label"]
                audio_label.insert_one({"audio_id": audio_id, 
                        "level": 2, 
                        "content": word,
                        "approve": 1})
        elif choose == 2:
            st.markdown(
                f"<h4 style='text-align: left; color: black;'>Sentence</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            st.text("content")
            st.write(sentence)
            check = ste.download_button('Download text', sentence) 
            if check:           
                fs = gridfs.GridFS(db)
                fs.put(file_uploader.getvalue(), filename=FILENAME)
                audio_file = db["fs.files"]
                update = {"$set": {"user_id": user_find_id}}
                audio_file.update_one({}, update)
                audio = audio_file.find_one({})
                audio_id = ObjectId(audio["_id"])
                audio_label = db["audio_label"]
                audio_label.insert_one({"audio_id": audio_id, 
                        "level": 1, 
                        "content": sentence,
                        "approve": 1})
        else:
            st.markdown(
                f"<h4 style='text-align: left; color: black;'>COMING SOON</h5>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        plt.close("all")


def load_audio_sample(file):
    y, sr = librosa.load(file, sr=22050)
    return y, sr

def action(file_uploader, selected_provided_file, transformations):
    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
    else:
        if selected_provided_file == "sample0":
            y, sr = librosa.load("data_wav/sample0.wav")
        elif selected_provided_file == "sample1":
            y, sr = librosa.load("samples/sample1.wav")
    
    choose_list = []
    for idx, ele in enumerate(transformations):
        if ele == True:
            choose_list.append(idx)
    
    plot_audio_transformations(y, sr, choose_list,file_uploader)


def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# VIETNAMESE RECOGNITION\n"
        "### Select the level prediction in the sidebar.\n"
        "Once you have chosen label techniques, select or upload an audio file\n. "
        'Then click "Apply" to start! \n\n'
    )
    placeholder2.markdown(
        "After clicking start, the individual steps of the pipeline are visualized. The ouput of the previous step is the input to the next step."
    )
    # placeholder.write("Create your audio pipeline by selecting augmentations in the sidebar.")
    st.sidebar.markdown("Choose the transformations here:")
    syllable = st.sidebar.checkbox("mức âm")
    word = st.sidebar.checkbox("mức từ")
    sentence = st.sidebar.checkbox("mức câu")
    clip_board = st.sidebar.checkbox("cắt ghép audio")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("(Optional) Upload an audio file here:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"]
    )
    st.sidebar.markdown("Or select a sample file here:")
    selected_provided_file = st.sidebar.selectbox(
        label="", options=["sample0", "sample1"]
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Apply"):
        global FILENAME
        # current_GMT = time.gmtime()
        # FILENAME = "{},{}".format(file_uploader.name,current_GMT);
        FILENAME = file_uploader.name
        placeholder.empty()
        placeholder2.empty()
        transformations = [
            syllable,
            word,
            sentence,
            clip_board
        ]
        print(transformations)
        action(
            file_uploader=file_uploader,
            selected_provided_file=selected_provided_file,
            transformations=transformations,
        )

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="vietnamese labeling")
    main()