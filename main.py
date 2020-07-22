import os
from glob import glob
import checkdir
import RecordAudio
import argparse
import clean
import removeSilence
import NormalizeSound
import DataAugmentation
import DataGenerator
import predict

import threading
import time

#ฟังก์ชันที่เช็คไฟล์โมเดลว่ามีหรือไม่
def checkFile(path):
    if os.path.exists(path) is True :
        x = os.listdir(path)
        if len(x)<3 :
            print("don't have model")
            return False
        else:
            print('have model')
            print(len(x))
            return True
    else :
        return False

#ฟังก์ชันที่ใช้สร้างไดเรกทอรี่ที่ใช้สำหรับเก็บไฟล์เสียงและวนลูปอัดเสียง
def mkdir_n_rec(headfile = None,mainpath = None, positive = None, Negative = None,printPositive = None,printNegative = None):
    mk = checkdir.mkpath(headfile,mainpath,positive,Negative)
    prt = [printPositive, printNegative]
    #ลูป 2 ครั้งเพื่อบันทึก Positive ,Negative ส่วนลูปในเพื่อบันทึกอย่างละ 5 ไฟล์
    for i in range(2):
        for j in range(5):
            print('กรุณาพูดคำว่า ',prt[i])
            time.sleep(0.25)
            RecordAudio.rec(mk[i])

#ฟังก์ชันที่เรียกใช้ clean เพื่อปรับขนาดข้อมูลให้พอดีกับ input ของโมเดล
def splitAudio(src = None, dst = None):
    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default=src,
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default=dst,
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=2.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    clean.split_wavs(args)

#ฟังก์ขันที่ใช้เทรนโมเดล
def data_Generator(path=None, saveModelpath=None):
    parser = argparse.ArgumentParser(description='Call for help Classification Training')
    parser.add_argument('--model_type',type=str, default='RNN_model',help='model to run lstm')
    parser.add_argument('--src_root',type=str,default=path)
    parser.add_argument('--dst_root', type=str, default=saveModelpath,help='Model path')
    parser.add_argument('--batch_size',type=int, default=32,help='batch_size') #ปรับ mini batch เพื่อแบ่งข้อมูลเป็นกลุ่มเล็กในการเทรนแต่ละ Epoch ข้อมูลจะน้อยลง ตวามเร็วจะเพิ่มขึ้น แนะนำให้ใช้ 1 เพื่อความแม่นยำ
    parser.add_argument('--delta_time','-dt',type=float,default=2.0,help='time in seconds to sample audio')
    parser.add_argument('--sample_rate','-sr',type=int,default=16000, help='sample rate')
    args, _ = parser.parse_known_args()
    DataGenerator.train(args)

#ฟังก์ชันที่เรียกใช้งานฟังก์ชันจำเป็นต่าง ๆ เมื่อเปิดใช้งานครั้งแรก
def firsttime():
    #อัดเสียงและสร้างโฟล์เดอร์
    print("don't have model ...")
    print('[Make Directory] Running ... ')
    checkdir.mkpath('Model')
    print('[Make Directory] Directory Model is Created')
    mkdir_n_rec('Audio','Help','True','False','"ช่วยด้วย"','อะไรก็ได้ เช่น "อาบน้ำ" "น้ำไม่ไหล" เป็นต้น หรือ ไม่พูด')
    print('[Make Directory] Directory and Record Help is Created')
    mkdir_n_rec('Audio','Confirm1','True','False','"ใข่"','"ไม่ใช่"')
    print('[Make Directory] Directory and Record Confirm1 is Created')
    mkdir_n_rec('Audio','Confirm2','True','False','"โอเค"','"ไม่ต้อง"')
    print('[Make Directory] Directory and Record Confirm2 is Created')
    print('[Make Directory] Finished... ')
    #checkdir('Audio','rmSilenceHelp','True','False')

    #ตัดเสียงเงียบของไฟล์เสียงที่อัด
    print('[Remove Silence in Audio] Running ... ')
    removeSilence.rmSilence('Audio/Help','Audio/rmSilenceHelp')
    removeSilence.rmSilence('Audio/Confirm1','Audio/rmSilenceConfirm1')
    removeSilence.rmSilence('Audio/Confirm2','Audio/rmSilenceConfirm2')
    print('[Remove Silence in Audio] Finished ... ')

    #Normalize Audio
    print('[Normalize Audio] Running ... ')
    NormalizeSound.normalizeAudio('Audio/rmSilenceHelp','Audio/normalizedHelp')
    NormalizeSound.normalizeAudio('Audio/rmSilenceConfirm1','Audio/normalizedConfirm1')
    NormalizeSound.normalizeAudio('Audio/rmSilenceConfirm2','Audio/normalizedConfirm2')
    print('[Normalize Audio] Finished ... ')

    #ปรับขนาดข้อมูลเสียงให้พอดีกับ input ของโมเดลก่อนทำ Data Augmentation เนื่องจากบางขั้นตอนต้องการไฟล์เสียงที่มีขนาดยาวขึ้นเช่น Shifting time หาก Audio มีขนาดสั้นการ Shift จะ Shift ไกลจนทำให้เสียงหาย
    print('[Split Audio] Running ... ')
    splitAudio('Audio/normalizedHelp','Audio/cleanHelp')
    splitAudio('Audio/normalizedConfirm1','Audio/cleanConfirm1')
    splitAudio('Audio/normalizedConfirm2','Audio/cleanConfirm2')
    print('[Split Audio] Finished ... ')

    #Data Augmentation
    print('[Data Augmentation] Running ... ')
    print('[Data Augmentation] Creating more Help Voice ... ')
    DataAugmentation.dataAugmentation('Audio/cleanHelp','Audio/dataAugmentationHelp')
    print('[Data Augmentation] Help Voice is Created... ')
    print('[Data Augmentation] Creating more Confirm1 Voice ... ')
    DataAugmentation.dataAugmentation('Audio/cleanConfirm1','Audio/dataAugmentationConfirm1')
    print('[Data Augmentation] Confirm1 Voice is Created... ')
    print('[Data Augmentation] Creating more Confirm2 Voice ... ')
    DataAugmentation.dataAugmentation('Audio/cleanConfirm2','Audio/dataAugmentationConfirm2')
    print('[Data Augmentation] Confirm2 Voice is Created... ')
    print('[Data Augmentation] Finished ... ')

    #ปรับขนาดข้อมูลเสียงให้พอดีกับ input ของโมเดลอีกครั้งหลังเนื่องจากการทำ Data Augmentation ในบางขั้นตอนทำให้ขนาดของไฟล์ Audio เปลี่ยนไป 
    print('[Split Audio] Running ... ')
    splitAudio('Audio/dataAugmentationHelp','Audio/cleanDataAug_Help')
    splitAudio('Audio/dataAugmentationConfirm1','Audio/cleanDataAug_Confirm1')
    splitAudio('Audio/dataAugmentationConfirm2','Audio/cleanDataAug_Confirm2')
    print('[Split Audio] Finished ... ')

    print('[Train Model] Running ... ')
    print('[Train Model] Training Help Model ... ')
    data_Generator('Audio/cleanDataAug_Help','Model/RNN_Help_Model.h5')
    print('[Train Model] Training Confirm1 Model ... ')
    data_Generator('Audio/cleanDataAug_Confirm1','Model/RNN_Confirm1_Model.h5')
    print('[Train Model] Training Confirm2 Model ... ')
    data_Generator('Audio/cleanDataAug_Confirm2','Model/RNN_Confirm2_Model.h5')
    print('[Train Model] Finished ... ')

#ฟังก์ขันเริ่มการทำนาย
def start_predict():
    ear = predict.SWHear()
    #แบ่ง task เป็น 2 task โดย task1 เพื่อทำงานเก็บค่าจากไมโครโฟนตลอดเวลา task2 เพื่อทำนาย 
    task1 = threading.Thread(target=ear.tape_add)
    task2 = threading.Thread(target=ear.predict)

    task1.start()
    task2.start()

    task1.join()
    task2.join()
   
    ear.close()
    print("DONE")
    


if __name__ == "__main__":
    x = checkFile('Model')
    if x == False:
        firsttime()
        start_predict()
    else:
        start_predict()
