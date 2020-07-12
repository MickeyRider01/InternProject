import pyaudio
import time
import pylab
import numpy as np
import librosa

from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from tensorflow.keras.models import load_model

import threading
import time

class SWHear(object):
    

    def __init__(self,device=None,startStreaming=True):
      

        self.chunk = 1024 # number of data points to read at a time
        self.rate = 16000 # time resolution of the recording device (Hz)
        self.state = 0
        self.predictHelp = False
        self.predictConfirm01 = False
        self.predictConfirm02 = False
        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength=2 #seconds
        self.tape=np.empty(self.rate*self.tapeLength)*np.nan
        
        self.model = load_model('Model/RNN_Help_Model.h5',
                                custom_objects={'Melspectrogram':Melspectrogram,
                                                'Normalization2D':Normalization2D})
        
        self.model_Confirm1 = load_model('Model/RNN_Confirm1_Model.h5',
                                        custom_objects={'Melspectrogram':Melspectrogram,
                                                        'Normalization2D':Normalization2D})

        self.model_Confirm2 = load_model('Model/RNN_Confirm2_Model.h5',
                                        custom_objects={'Melspectrogram':Melspectrogram,
                                                        'Normalization2D':Normalization2D})

        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()
    


    def stream_read(self):
        """return values for a single chunk"""
        data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
        #print(data)
        return data

    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream=self.p.open(format=pyaudio.paInt16,channels=1,
                                rate=self.rate,input=True,
                                frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk]=self.tape[self.chunk:]
        self.tape[-self.chunk:]=self.stream_read()

    def tape_forever(self,plotSec=.25):
        t1=0
 
        try:
            while True:
                self.tape_add()
                if (time.time()-t1)>plotSec:
                    t1=time.time()                 
        except:
            print(" ~~ exception (keyboard?)")
            return

    def tape_plot(self,saveAs="03.png"):
        """plot what's in the tape."""
        pylab.plot(np.arange(len(self.tape))/self.rate,self.tape)
        #pylab.plot(self.tape)
        temp = np.arange(len(self.tape))/self.rate,self.tape
        
        pylab.axis([0,self.tapeLength,-2**16/2,2**16/2])
        #np.save("Cells",self.tape)
        if saveAs:
            t1=time.time()
            pylab.savefig(saveAs,dpi=50)
            print("plotting saving took %.02f ms"%((time.time()-t1)*1000))
        else:
            pylab.show()
            print() #good for IPython
        pylab.close('all')


    def predict(self):
        #self.tape_add()
        while True:
            pre = self.tape
            #print('predict help : ',pre)
            print('หากต้องการความช่วยเหลือกรุณาพูด "ช่วยด้วย" ')
            pre = pre.reshape(1,-1)
            score = self.model.predict(pre,verbose=1)
            label_index=np.argmax(score)
            if label_index == 1 :
                self.predictHelp = True
                print('>>>>>>>>>>>>>>>>>>>>>>>   Help')
                #time.sleep(15)

                # Confirm 1

                if self.predictHelp == True :
                    print('[start first confirm in 3s] ...')
                    time.sleep(1)
                    print('[start first confirm in 2s] ...')
                    time.sleep(1)
                    print('[start first confirm in 1s] ...')
                    time.sleep(1)
                    print('คุณต้องการความช่วยเหลือ "ใช่" หรือ "ไม่" ')
                    time.sleep(2)
                    pre1 = self.tape
                    #print("predict yes or no : ",pre1)
                    pre1 = pre1.reshape(1,-1)
                    score1 = self.model_Confirm1.predict(pre1,verbose=1)
                    label_index1=np.argmax(score1)
                    if label_index1 == 1 :
                        self.predictConfirm01 = True
                        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Yes')
                        #time.sleep(10)

                        # Confirm 2

                        if self.predictConfirm01 == True :
                            print('[start second confirm in 3s] ...')
                            time.sleep(1)
                            print('[start second confirm in 2s] ...')
                            time.sleep(1)
                            print('[start second confirm in 1s] ...')
                            time.sleep(1)
                            print('กรุณาพูด "โอเค" เพื่อยืนยันการส่งข้อความแจ้งเตือน  หากไม่ต้องการ กรุณาพูด "ไม่ต้อง" ')
                            time.sleep(2)
                            pre2 = self.tape
                            #print('predict okey or do not',pre2)
                            pre2 = pre2.reshape(1,-1)
                            score2 = self.model_Confirm2.predict(pre2,verbose=1)
                            label_index2=np.argmax(score2)
                            if label_index2 == 1 :
                                self.predictConfirm02 = True
                                print('[sending messege] "Please Help"')
                                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Okey')
                                time.sleep(5)
                            else :
                                self.predictConfirm02 = False 
                                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   Do not')
                                time.sleep(5)      

                    else :
                        self.predictConfirm01 = False
                        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   No')
                        time.sleep(1)      

            else :
                self.predictHelp = False
                print('>>>>>>>>>>>>>>>>>>>>>>>   No problem')


if __name__=="__main__":
    ear=SWHear()

    task1 = threading.Thread(target=ear.tape_forever)
    task2 = threading.Thread(target=ear.predict)

    task1.start()
    task2.start()

    task1.join()
    task2.join()

    ear.close()
    print("DONE")