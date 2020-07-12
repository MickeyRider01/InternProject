from pydub import AudioSegment
import librosa
import os
import checkdir
from glob import glob
# positive_path = 'NewData/Data/True/'
# negative_path = 'NewData/Data/False/'
# save_path = 'NewData/rm_silence/True/'
# save_path_negative = 'NewData/rm_silence/False/'

# positive_path = 'NewData/Confirm2/Ok/'
# negative_path = "NewData/Confirm2/Don't/"
# save_path = 'NewData/rm_silence_Confirm2/Ok/'
# save_path_negative = "NewData/rm_silence_Confirm2/Don't/"

# test_path = 'NewTest/Help/'
# test_path_negative = 'NewTest/Negative/'

# save_test_path = 'rm_sl_newtest/Help/'
# save_test_path_negative = 'rm_sl_newtest/Negative/'



def detect_leading_silence(sound, silence_threshold=-45.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def rm(path, save_path):
    data = os.listdir(path)
    for i in data :
        sound = AudioSegment.from_file(path+'/'+i, format="wav")

        # sound = librosa.core.load("Data/CallForHelp/04.wav")[0]

        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        #librosa.output.write_wav('test_rm_silence01.wav',trimmed_sound,16000)
        trimmed_sound.export(save_path+'/'+i, format="wav")

def rmSilence(path = None, save_path = None):
    data = os.listdir(path)
    wav_paths = glob('{}/**'.format(path), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    
    checkdir.check_dir(save_path)
    classes = os.listdir(path)
    for i in classes:
        target_dir = os.path.join(save_path, i)
        checkdir.check_dir(target_dir)
        src_dir = os.path.join(path, i)
        rm(src_dir,target_dir)
    