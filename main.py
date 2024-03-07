from sound_c import read_sound_folder_csv, apply_mfcc_to_folder


folder_path = 'D:/HUST/dev/py/sound_classification/audio_train'
csv_file_path = 'D:/HUST/dev/py/sound_classification/bee_sound.csv'
sound_data_csv = read_sound_folder_csv(folder_path, csv_file_path)

#test read sound folder
if sound_data_csv is not None:
    print(sound_data_csv)

mfcc_data = apply_mfcc_to_folder(folder_path)
#test MFCCs
if mfcc_data is not None:
    print("MFCC Data:")
    print(mfcc_data)