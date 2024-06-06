import torch
import tensorflow as tf

from src.ExtractFrameData import write_frame_data


def do_stuff():
    write_frame_data("data/survey/1c069c5e-82f5-46a0-80eb-b9daa63bd3b2", "Anger_0_1310.webm")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)
    print("Cuda Devices:", torch.cuda.device_count())
    print("Num GPUs Available:", tf.config.list_physical_devices())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
