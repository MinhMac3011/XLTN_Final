# Import necessary libraries
from pydub import AudioSegment

import pydub
pydub.AudioSegment.ffmpeg = "E:\Documents\Desktop\XLTN_FinalTerm/ffmpeg-4.3-win32-static/bin/ffmpeg.exe"

AudioSegment.converter = "E:\Documents\Desktop\XLTN_FinalTerm/ffmpeg-4.3-win32-static/bin/ffmpeg.exe" #thu vien fix loi mot so file wav ko doc duoc
AudioSegment.ffmpeg = "E:\Documents\Desktop\XLTN_FinalTerm/ffmpeg-4.3-win32-static/bin/ffmpeg.exe"
AudioSegment.ffprobe = "E:\Documents\Desktop\XLTN_FinalTerm/ffmpeg-4.3-win32-static/bin/ffprobe.exe"

# pip install pydub
# pip install pysrt


import pysrt

import os
from pathlib import Path
import pathlib
import glob

def convert_to_millisecond(miniSub):

    start =   miniSub.start.ordinal
    end   =   miniSub.end.ordinal

    return start, end


if __name__ == '__main__':

    word_dict = dict()

    dirName = "E:\Documents\Desktop\XLTN_FinalTerm\Data2" #todo : sua thanh foler muon chua data da xu ly

    print("Start processing ......")
    
    #count = 0

    for wav_file in glob.iglob('E:\Documents\Desktop\XLTN_FinalTerm\Data\Ex1-Dataset - Copy/**/*.wav'): #todo: neu muon thay "D:/Ex1-Dataset" bang folder source tai ve.

        srt_file = wav_file[:wav_file.find(".")] + ".srt"


        audio = AudioSegment.from_wav(wav_file)
        #
        subs = pysrt.open(srt_file)
        #
        total_subs = len(subs)
        #
        #
        #
        for i in range(0, total_subs):


            start, end = convert_to_millisecond(subs[i])

            word_sound = audio[start:end]

            word_content = subs[i].text.lower()

            word_content = word_content.replace(",", "") #phong khi ai do quen khong xoa cac ki tu dac biet
            word_content = word_content.replace(".", "")
            word_content = word_content.replace("\"", "")
            word_content = word_content.replace("(", "")
            word_content = word_content.replace(")", "")
            word_content = word_content.replace("+", "")
            word_content = word_content.replace("-", "")
            word_content = word_content.replace("*", "")
            word_content = word_content.replace(":", "")
            word_content = word_content.replace("%", "")
            word_content = word_content.replace(":", "")

            if('\n' in word_content):      #ignore those cases have forms in srt file: start ---> end : word1 \n word2 \n word3 ... (TH nay la do sap de len nhau)
                continue


            if word_content == 'con': #ko hieu tu nay khong xu ly dc nen bo qua
                continue

            if word_content in word_dict:

                id = word_dict[word_content]

                word_dict[word_content] += 1

                full_dir_path = os.path.join(dirName, word_content)


                #pathlib.Path(full_dir_path).mkdir(parents=True, exist_ok=True)

                file_name = os.path.join(full_dir_path, word_content + str(id) + '.wav')

                word_sound.export(file_name, format="wav")



            else:
                word_dict[word_content] = 1

                full_dir_path = os.path.join(dirName, word_content)

                pathlib.Path(full_dir_path).mkdir(parents=True, exist_ok=True)

                file_name = os.path.join(full_dir_path, word_content + '.wav')

                word_sound.export(file_name, format="wav")

        print(" processed ", wav_file, " file ......")
        #count += 1

    print("Finish collecting data ........!")
