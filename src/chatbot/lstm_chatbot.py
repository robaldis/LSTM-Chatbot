import data_setup
import model
import tensorflow as tf
import os

import re
import random
import time

from sklearn.model_selection import train_test_split

def refactoring_data():
    data_path = "human_text.txt"
    data_path2 = "robot_text.txt"
    # Defining lines as a list of each line
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    with open(data_path2, 'r', encoding='utf-8') as f:
        lines2 = f.read().split('\n')
    lines = [re.sub(r"\[\w+\]",'hi',line) for line in lines]
    lines = [" ".join(re.findall(r"\w+",line)) for line in lines]
    lines2 = [re.sub(r"\[\w+\]",'',line) for line in lines2]
    lines2 = [" ".join(re.findall(r"\w+",line)) for line in lines2]
    # grouping lines by response pair
    pairs = list(zip(lines,lines2))
    #random.shuffle(pairs)
    with open ('human-robot.txt', 'w+') as f:
        for line in pairs:
            f.writelines(line[0] + '\t' + line[1] + '\n')





# refactoring_data()
path_to_file='assets/human-robot.txt'

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

# path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

input_tensor, target_tensor, input_token, target_token = data_setup.load_dataset(path_to_file, 2000)


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
convert(target_token, target_tensor[0])




# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(input_token.word_index)+1
vocab_tar_size = len(target_token.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

EPOCHS = 10


model = model.Model(dataset, vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, target_token)
model.train(EPOCHS, steps_per_epoch)