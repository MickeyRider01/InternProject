import os
from glob import glob


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def mkpath(headfile = None,mainpath = None, positive = None, Negative = None):
    classes = [positive,Negative]
    AudioFolder = headfile
    part_ap = []
    check_dir(AudioFolder)
    if mainpath != None :
        mainFolder = mainpath
        target1 = os.path.join(AudioFolder,mainFolder)
        check_dir(target1)
        if positive != None and Negative != None:
            for i in classes:
                target2 = os.path.join(target1,i)
                check_dir(target2)  
                part_ap.append(target2)
                print(target2)

    return part_ap

if __name__ == "__main__":

    x = mkpath('Audio','Confirm1','Yes','No')
    
    print(x)
