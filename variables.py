sample_rate = 16000
signal_length = sample_rate

frame_size = 1024
hop_length = frame_size // 2
n_mels = 64

learning_rate = 0.001
epochs = 20
batch_size = 32

# Data Paths
import os

clean_train_dir = 'data/train/clean'
noisy_train_dir = 'data/train/noisy'

clean_test_dir  = 'data/test/clean'
noisy_test_dir  = 'data/test/noisy'

serialized_train_dir = 'data/serialized_train_data'
serialized_test_dir = 'data/serialized_test_data'

if not os.path.exists(serialized_train_dir):
    os.makedirs(serialized_train_dir)

if not os.path.exists(serialized_test_dir):
    os.makedirs(serialized_test_dir)

window_size = 2 ** 14  # about 1 second of samples
stride = 0.5
negative_slope = 0.03