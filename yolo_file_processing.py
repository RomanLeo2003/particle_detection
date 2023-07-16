import os
from shutil import copyfile

train_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\PycharmProjects\particle_detection\labels_move\train")]
valid_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\PycharmProjects\particle_detection\labels_move\valid")]
test_dir = [file[:-4] for file in os.listdir(r"C:\Users\user\PycharmProjects\particle_detection\labels_move\test")]
resized_path = r"C:\Users\user\PycharmProjects\particle_detection\resized_images"

for filename in os.listdir(r"C:\Users\user\PycharmProjects\particle_detection\resized_images"):
    if filename[:-4] in train_dir:
        copyfile(resized_path + "/" + filename, rf'C:\Users\user\PycharmProjects\particle_detection\labels_move\train\{filename}')
    elif filename[:-4] in valid_dir:
        copyfile(resized_path + "/" + filename, rf'C:\Users\user\PycharmProjects\particle_detection\labels_move\valid\{filename}')
    elif filename[:-4] in test_dir:
        copyfile(resized_path + "/" + filename, rf'C:\Users\user\PycharmProjects\particle_detection\labels_move\test\{filename}')