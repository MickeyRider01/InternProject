from pydub import AudioSegment, effects
import os
import checkdir

# positive_path = 'NewData/rm_silence/True/'
# negative_path = 'NewData/rm_silence/False/'
# save_path_positive = 'NewData/Normalized/True/'
# save_path_negative = 'NewData/Normalized/False/'

# positive_path = 'NewTest/True/'
# negative_path = 'NewTest/False/'
# save_path_positive = 'NewData/Normalized_Test/True/'
# save_path_negative = 'NewData/Normalized_Test/False/'

positive_path = 'NewData/rm_silence_Confirm/Yes/'
negative_path = "NewData/rm_silence_Confirm/No/"
save_path_positive = 'NewData/Normalized_Confirm/Yes/'
save_path_negative = "NewData/Normalized_Confirm/No/"

#data = os.listdir(negative_path)
def normalized(path = None, save_path = None):
    data = os.listdir(path)
    for i in data :
        sound = AudioSegment.from_file(path+'/'+i)
        normalize = effects.normalize(sound)
        normalize.export(save_path+'/'+i,format='wav')

def normalizeAudio(path = None, save_path = None):
    data = os.listdir(path)
    #wav_paths = glob('{}/**'.format(path), recursive=True)
    #wav_paths = [x for x in wav_paths if '.wav' in x]
    
    checkdir.check_dir(save_path)
    classes = os.listdir(path)
    for i in classes:
        target_dir = os.path.join(save_path, i)
        checkdir.check_dir(target_dir)
        src_dir = os.path.join(path, i)
        normalized(src_dir,target_dir)