from pydub import AudioSegment
import librosa
import os
import checkdir
from glob import glob

# ฟังก์ชันที่ใช้ในการหาเสียงที่ไม่ใช่เสียงเงียบว่าอยู่ใน ms ที่เท่าไหร่
def detect_leading_silence(sound, silence_threshold=-45.0, chunk_size=10):

    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms
# ฟังก์ชันที่ใช้ในการตัดเสียงเงียบออกไป จากการหาว่า ms ที่มีเสียงคือ ms ที่เท่าไหร่ หัว และท้ายไฟล์ Audio จากนั้นจะตัดส่วนนั้นมาบันทึก
def rm(path, save_path):
    data = os.listdir(path)
    for i in data :
        sound = AudioSegment.from_file(path+'/'+i, format="wav")

        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)    
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export(save_path+'/'+i, format="wav")

# ฟังก์ชันที่ใช้เช็ก path ต่าง ๆ และสร้างไฟล์ให้ path ที่บันทึก เมื่อเรียกใช้ removeSilence.py จะเรียกกับฟังก์ชันนี้โปรแกรมจะทำงานทุกขั้นตอนของการลบเสียงเงียบ
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
    