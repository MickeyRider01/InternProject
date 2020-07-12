from pydub import AudioSegment, effects
import os
import checkdir

def normalized(path = None, save_path = None):
    data = os.listdir(path)
    for i in data :
        sound = AudioSegment.from_file(path+'/'+i)
        normalize = effects.normalize(sound)
        normalize.export(save_path+'/'+i,format='wav')

def normalizeAudio(path = None, save_path = None):
    data = os.listdir(path)
    
    
    checkdir.check_dir(save_path)
    classes = os.listdir(path)
    for i in classes:
        target_dir = os.path.join(save_path, i)
        checkdir.check_dir(target_dir)
        src_dir = os.path.join(path, i)
        normalized(src_dir,target_dir)