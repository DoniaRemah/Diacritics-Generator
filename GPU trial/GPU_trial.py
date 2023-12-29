import tensorflow as tf

# Check if there are available GPUs
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Check if TensorFlow is using GPU for operations
print("TensorFlow is using GPU: ", tf.test.is_gpu_available())

import torch

# Check if there are available GPUs
print("Num GPUs Available: ", torch.cuda.device_count())

# Check if PyTorch is using GPU for operations
print("PyTorch is using GPU: ", torch.cuda.is_available())
