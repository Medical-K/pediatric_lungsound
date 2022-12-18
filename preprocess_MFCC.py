import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import os
import json
import numpy as np
import pandas as pd
from pydub import AudioSegment
from skimage.restoration import denoise_wavelet
import pdb

#########################
OUTPUT_SECONDS = 6      # What time do you want to make for the same output?
n_fft = 660
n_mfcc = 40
hop_length = 512
sr = 22050

#########################
def save_mfcc():
    ## dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    thinning_data_crackle = {
        "name" : [],
        "mfcc": [],
        "labels": []
    }

    thinning_data_wheezing = {
        "name" : [],
        "mfcc": [],
        "labels": []
    }    

    thinning_data_normal = {
        "name" : [],
        "mfcc": [],
        "labels": []
    }        

    #############################################################
    ## Set classes
    label_classes = ["crackle", "wheezing", "normal"]
    print('Classes: ', label_classes)
    data["mapping"] = label_classes
    #############################################################
    print('=======================================')
    #############################################################
    # Final_labeling
    final_sheet = pd.read_csv('Final_ICBHI_2022_pediatric_by_K.csv')
    #############################################################
    for N in range(final_sheet.shape[0]):
        ## load audio file
        filename = final_sheet['filename'][N]
        filepath = os.path.join("ICBHI_final_database", filename)

        ## define label
        label = final_sheet['lung'][N]
        print("Signal {}/ label = {}".format(filename, label))

        ## extract signal
        signal, sr = librosa.load(filepath, sr=None)
        SAMPLES_PER_TRACK = signal.shape[0]
        print("* Signal samples:{}, duration:{}, label:{}".format(SAMPLES_PER_TRACK, round(SAMPLES_PER_TRACK/sr, 2), label))

        ## slicing the signal
        start = final_sheet['start'][N] * sr
        end = final_sheet['end'][N] * sr
        signal = signal[int(start):int(end)]
        SAMPLES_PER_TRACK = signal.shape[0]
        print("* [SLICING] Signal samples:{}, duration:{}, label:{}".format(SAMPLES_PER_TRACK, round(SAMPLES_PER_TRACK/sr, 2), label))

        if int(SAMPLES_PER_TRACK) == 0:
            pdb.set_trace()

        ## Denoising
        signal = denoise_wavelet(signal, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')

        ## repeat if samples is shorter than 6 seconds (criteria: max sampling(=44100))
        if SAMPLES_PER_TRACK < 44100 * 6:
            if (SAMPLES_PER_TRACK > 44100 * 3):
                signal = np.hstack((signal, signal))        
            elif (SAMPLES_PER_TRACK <= 44100 * 3) and (SAMPLES_PER_TRACK > 44100 * 2):
                signal = np.hstack((signal, signal, signal))
            elif (SAMPLES_PER_TRACK <= 44100 * 2) and (SAMPLES_PER_TRACK > 44100 * 1.5):
                signal = np.hstack((signal, signal, signal, signal))
            elif (SAMPLES_PER_TRACK <= 44100 * 1.5) and (SAMPLES_PER_TRACK > 44100 * 1.2):
                signal = np.hstack((signal, signal, signal, signal, signal))
            elif (SAMPLES_PER_TRACK <= 44100 * 1.2) and (SAMPLES_PER_TRACK > 44100):
                signal = np.hstack((signal, signal, signal, signal, signal, signal))
            elif (SAMPLES_PER_TRACK <= 44100) and (SAMPLES_PER_TRACK > 20000):
                signal = np.hstack((signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal))
            elif (SAMPLES_PER_TRACK <= 20000) and (SAMPLES_PER_TRACK > 10000):
                signal = np.hstack((signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal))                                    
            elif (SAMPLES_PER_TRACK <= 10000) and (SAMPLES_PER_TRACK > 5000):
                signal = np.hstack((signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal))            
            else:
                signal = np.hstack((signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,\
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, 
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,   
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,                                                                    
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, 
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,   
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal,                                                                                                        
                                    signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal, signal))
            print('    shorter than 6seconds -> repeat signal!')

        ## cut 6 seconds
        signal = signal[:44100*6]    

        ## Make the whole signal into same length
        print("* Final Signal samples:{}, duration:{}, label:{}".format(signal.shape[0], round(signal.shape[0]/sr, 2), label))

        #D = np.abs(librosa.stft(signal))**2
        #S = librosa.feature.melspectrogram(S=D, sr=sr)
        #print('Mel spectrogram shape: ', S.shape)

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        print('MFCC shape: ', mfcc.shape)
        #data["mfcc"].append(mfcc.tolist())

        ## Append
        #data["labels"].append(label)

        # For thinning_dataset
        if label == 'crackle':
            thinning_data_crackle['name'].append(filename.split('.')[0].split('/')[-1])
            thinning_data_crackle['mfcc'].append(mfcc.tolist())
            thinning_data_crackle['labels'].append(label)
        elif label == 'wheezing':
            thinning_data_wheezing['name'].append(filename.split('.')[0].split('/')[-1])
            thinning_data_wheezing['mfcc'].append(mfcc.tolist())
            thinning_data_wheezing['labels'].append(label)
        elif label == 'normal':
            thinning_data_normal['name'].append(filename.split('.')[0].split('/')[-1])
            thinning_data_normal['mfcc'].append(mfcc.tolist())
            thinning_data_normal['labels'].append(label)            
        else:
            raise Exception('Label is not correct!')        
        print("{} finished!, label: {}".format(filepath, label))                
        print("============================================")

        try:
            if (len(thinning_data_crackle['mfcc']) == 0) or (len(thinning_data_wheezing['mfcc'])==0):
                pass
            else:
                a = np.array(thinning_data_normal['mfcc']).shape[2]
                b = np.array(thinning_data_crackle['mfcc']).shape[2]
                c = np.array(thinning_data_wheezing['mfcc']).shape[2]
        except:
            pdb.set_trace()

    # with open(JSON_PATH, "w") as fp:
    #     json.dump(data, fp, indent=4)
    # print("Save finished!")

    with open('./thinning_normal_MFCC_ICBHI_pediatric.json', 'w') as f:
        json.dump(thinning_data_normal, f, indent=4)
    print("Thinning normal data finished!")  

    with open('./thinning_crackle_MFCC_ICBHI_pediatric.json', 'w') as f:
        json.dump(thinning_data_crackle, f, indent=4)
    print("Thinning crackle data finished!")    

    with open('./thinning_wheezing_MFCC_ICBHI_pediatric.json', 'w') as f:
        json.dump(thinning_data_wheezing, f, indent=4)
    print("Thinning wheezing data finished!")     

if __name__ == "__main__":
    save_mfcc()