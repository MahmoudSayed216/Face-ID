import zipfile
import os
import math
import json

class label:
    def __init__():
        pass


def extract_data(zip_file_path: str, output_dir):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


    return os.path.exists(output_dir)
        



def create_data_env_in_yolo_format(base_dir):
    DATA_DIR_PATH            = os.path.join(base_dir, "retinaface")
    #__
    IMAGES_SUBDIR_PATH       = os.path.join(DATA_DIR_PATH, "images")
    LABELS_SUBDIR_PATH       = os.path.join(DATA_DIR_PATH, "labels")
    #__
    TRAIN_IMAGES_SUBDIR_PATH = os.path.join(IMAGES_SUBDIR_PATH, "train")
    VAL_IMAGES_SUBDIR_PATH   = os.path.join(IMAGES_SUBDIR_PATH, "val")
    TEST_IMAGES_SUBDIR_PATH  = os.path.join(IMAGES_SUBDIR_PATH, "test")
    #__
    TRAIN_LAEBLS_SUBDIR_PATH = os.path.join(LABELS_SUBDIR_PATH, "train")
    VAL_LAEBLS_SUBDIR_PATH   = os.path.join(LABELS_SUBDIR_PATH, "val")
    TEST_LAEBLS_SUBDIR_PATH  = os.path.join(LABELS_SUBDIR_PATH, "test")

    paths = [DATA_DIR_PATH, 
              IMAGES_SUBDIR_PATH, 
              LABELS_SUBDIR_PATH, 
              TRAIN_IMAGES_SUBDIR_PATH, 
              VAL_IMAGES_SUBDIR_PATH, 
              TEST_IMAGES_SUBDIR_PATH,
              TRAIN_LAEBLS_SUBDIR_PATH,
              VAL_LAEBLS_SUBDIR_PATH,
              TEST_LAEBLS_SUBDIR_PATH
              ]

    try:
        for path in paths:
            os.makedirs(name= path, exist_ok= False)
    except:
        return False


    return True

def reorganize_raw_labels(extracted_raw_labels_base_dir, destination_dir, split_ratios):
    def parse_line(line: str):
        values = line.split(' ')

        x   = float(values[0])
        y   = float(values[1])
        w   = float(values[2])
        h   = float(values[3])
        l1x = float(values[4])
        l1y = float(values[5])
        l1v = float(values[6])
        l2x = float(values[7])
        l2y = float(values[8])
        l2v = float(values[9])
        l3x = float(values[10])
        l3y = float(values[11])
        l3v = float(values[12])
        l4x = float(values[13])
        l4y = float(values[14])
        l4v = float(values[15])
        l5x = float(values[16])
        l5y = float(values[17])
        l5v = float(values[18])
        l1v = (l1v == 0.0) or (l1v == 1.0)
        l2v = (l2v == 0.0) or (l2v == 1.0)
        l3v = (l3v == 0.0) or (l3v == 1.0)
        l4v = (l4v == 0.0) or (l4v == 1.0)
        l5v = (l5v == 0.0) or (l5v == 1.0)

        return (x, y, w, h, l1x, l1y, l1v, l2x, l2y, l2v, l3x, l3y, l3v, l4x, l4y, l4v, l5x, l5y, l5v)


        

    TRAIN_TXT_PATH  = os.path.join(extracted_raw_labels_base_dir, "train", "label.txt")

    if not math.isclose(sum(split_ratios), 1.0, abs_tol=0.0001):
        return False

    file = open(TRAIN_TXT_PATH)
    lines = file.readlines()
    file.close()
    # "# 0--Parade/0_Parade_marchingband_1_849.jpg"
    # "449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82"
    data: dict[str, list[str]] = {}
    image = ""
    for line in lines:
        if "#" in line:
            image = line[2:]
            data[image] = []
        else:
            line = parse_line(line)
            print(line)
            data[image].append(line)
            break


    



def main():
    BASE_DIR = "../data"
    # BASE_RAW_LABELS_DIR = "../data/raw_labels"
    BASE_RAW_LABELS_DIR = os.path.join(BASE_DIR, "raw_labels")
    print("BASE RAW LABELS DIR:", BASE_RAW_LABELS_DIR)
    ZIP_PATH = os.path.join(BASE_RAW_LABELS_DIR, "retinaface_gt_v1.1.zip")
    FILE_NAME = "retinaface_gt_v1.1"
    OUTPUT_DIR = os.path.join(BASE_RAW_LABELS_DIR, FILE_NAME)
    success = extract_data(zip_file_path= ZIP_PATH, output_dir= OUTPUT_DIR)
    if not success:
        print("Error with extracting data")
        exit(1)
    else:
        print("Data extracted successfully")



    DATA_DST_BASE = os.path.join(BASE_DIR, FILE_NAME)
    success = create_data_env_in_yolo_format(DATA_DST_BASE)
    if not success:
        print("Error with creating directories")
        exit(1)
    else:
        print("Data environment in yolo format created successfully")



    success = reorganize_raw_labels(extracted_raw_labels_base_dir= OUTPUT_DIR, destination_dir= DATA_DST_BASE, split_ratios=[0.6, 0.1, 0.3])
    if not success:
        print("Error while reorganizing labels")
        exit()
    else:
        print("Data re-organized successfully")




if __name__ == "__main__":
    main()