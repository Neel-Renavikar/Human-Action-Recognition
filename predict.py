import csv
import glob
import os
import os.path
from subprocess import call
import sys

data_file = []
frames_considered = 20
#folder = 'test_data'
folder = str(sys.argv[1])
#print(folder)
test_files = glob.glob(os.path.join(folder , '*.avi'))
test_files = sorted(test_files)
for video_file in test_files:
    parts = video_file.split(os.path.sep)
    filename = parts[1]
    filename_no_ext = filename.split('.')[0]
    if not os.path.exists(os.path.join(folder, filename_no_ext + 't')):
        os.makedirs(os.path.join(folder, filename_no_ext + 't'))
    src = os.path.join(folder , filename)
    dest = os.path.join(folder ,filename_no_ext + 't', filename_no_ext + '-%04d.jpg')
    call(['ffmpeg', "-i", src, dest])
    generated_files = glob.glob(os.path.join(folder, filename_no_ext + 't', filename_no_ext + '*.jpg'))
    number_frames = len(generated_files)
    data_file.append([filename_no_ext, number_frames])
#print(data_file)

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np
from tqdm import tqdm

seq_model = InceptionV3(weights='imagenet', include_top=True)
from keras.preprocessing import image

pbar = tqdm(total = len(data_file))
if not os.path.exists(os.path.join(folder, 'sequencesm')):
    os.makedirs(os.path.join(folder, 'sequencesm'))
for item in data_file:
    output_frames = []
    path = os.path.join(folder, 'sequencesm', item[0] + '-' + str(frames_considered) + '-features')
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue
    else:
        os.mknod(path+'.npy')
    frames_path = os.path.join(folder, item[0] + 't')
    file_name = item[0]
    frames = sorted(glob.glob(os.path.join(frames_path, file_name + '*.jpg')))
    if (int)(item[1]) < frames_considered:
        reps = (int)(frames_considered/len(frames))
        remainder = (frames_considered%len(frames))
        startreps = reps + (int)(remainder/2)
        endreps = reps + (int)(remainder - startreps + reps)
        frame_number  = 0
        #print(str(reps) + " " +str(remainder) + " " + str(startreps) + " " + str(endreps))
        c = 0
        for frame in frames:
            if frame_number == 0:
                for i in range(0, startreps):
                    output_frames.append(frame)
                    c += 1
                    print(c)
            elif frame_number == (len(frames) - 1):
                for i in range(0, endreps):
                    output_frames.append(frame)
                    c += 1
                    print(c)
            else:
                for i in range(0,reps):
                    output_frames.append(frame)
                    c += 1
                    print(c)
            frame_number += 1
    else:
        skip = len(frames) // frames_considered
        output_frames = [frames[i] for i in range(0, len(frames), skip)]
    sequence = []
    count = 0
    #print(item[0] + str(len(output_frames)))
    for frame in output_frames:
        if count < frames_considered:
            img = image.load_img(frame, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)
            x = preprocess_input(x)
            features = seq_model.predict(x)
            features = features[0]
            count += 1
            sequence.append(features)
    np.save(path, sequence)
    pbar.update(1)
pbar.close()

from keras.models import load_model
model = load_model('90-10.h5')
data_file1 = []
classes = ['Golf Swing', 'Kicking', 'Lifting', 'Riding Horse', 'Running', 'SkateBoarding', 'Swing-Bench', 'Swing-Side', 'Walking']
predicted_result = []
for item in data_file:
    path = os.path.join(folder,'sequencesm',item[0]+'-' + str(frames_considered) + '-features.npy')
    if os.path.isfile(path):
        sequence = np.load(path)
    np.reshape(sequence, (sequence.shape[0], 1, sequence.shape[1]))
    sequence = sequence.reshape(1, frames_considered, 1000)
    print(item[0] + '.avi ' + classes[int(model.predict_classes(sequence))],file=open("output.txt", "a"))
