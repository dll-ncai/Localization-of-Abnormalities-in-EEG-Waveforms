import random
import os 

random.seed(444)

'''

This code produces a txt file that contains information of each window that was 
transformed into a cwt image and used in the deep learning dataset. This is done
to keep track of each transformed window and use it to duplicate the training,
validation and most importantly test set. Information is stored in the following
format: '[EDF file name] [window channel] [window number] [label]'. These are 
seperated using spaces. 

'''

PATH = '/home/Desktop/final_dataset_21st_Sep/Final/'
train_path = PATH + 'train/'
test_path = PATH + 'test/'
valid_path = PATH + 'valid/'

save_path = '/home/Downloads/FYP - Notebooks/'

def first_line(path):
    var = path.split('/')[-2]
    return var

def label_conv(folder_name):
    if folder_name == 'normal':
        return 0
    elif folder_name == 'slowing-wave':
        return 1
    elif folder_name == 'spike-and-wave':
        return 2
    else:
        print('Error in Label')
        return None

def name_changer(file_name, folder_name):
    file_name_list = file_name.split('_')
    if 'img' in file_name_list[0]:
        window = file_name_list[1]
        channel = file_name_list[2]
        edf_file = file_name_list[3][:-4]
        label = str(label_conv(folder_name))
    else:
        window = file_name_list[2]
        channel = file_name_list[3]
        edf_file = file_name_list[4][:-4]
        label = str(label_conv(folder_name))

    edf_file = str(10000000 + int(edf_file))[1:] + '.edf'

    return edf_file, channel, window, label

if __name__ == '__main__':
    paths = [train_path, valid_path, test_path]
    with open(save_path + 'window_tracker.txt', 'w') as f:   
        for single_path in paths:
            f.write(first_line(single_path))
            f.write('\n')
            folder_list = os.listdir(single_path)
            for folder in folder_list:
                for file_name in os.listdir(single_path + folder):
                    edf_file, channel, window, label = name_changer(file_name, folder)
                    f.write(edf_file + ' ' + channel + ' ' + window + ' ' + label)
                    f.write('\n')
            f.write('\n')
