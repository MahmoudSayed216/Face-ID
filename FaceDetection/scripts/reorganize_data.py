import zipfile
import os
import math
import json
import sys
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split



def check_success_state(success, fail_message, pass_message):
    if not success:
        print(fail_message)
        exit(1)
    else:
        print(pass_message)


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

    paths = [
                DATA_DIR_PATH, 
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
        return False, None, None


    return True, IMAGES_SUBDIR_PATH, LABELS_SUBDIR_PATH

def obtain_data(extracted_raw_labels_base_dir):
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

        # x_c_normal = (x+w/2)/w # WRONG, DIVIDE BY IMAGE WIDTH
        # y_c_normal = (y+h/2)/h # WRONG, DIVIDE BY IMAGE HEIGHT

        return (x, y, w, h, l1x, l1y, l1v, l2x, l2y, l2v, l3x, l3y, l3v, l4x, l4y, l4v, l5x, l5y, l5v)


    try: 

        TRAIN_TXT_PATH  = os.path.join(extracted_raw_labels_base_dir, "train", "label.txt")

        

        file = open(TRAIN_TXT_PATH)
        lines = file.readlines()
        file.close()
        # "# 0--Parade/0_Parade_marchingband_1_849.jpg"
        # "449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82"
        data: dict[str, list[str]] = {}
        image = ""
        x = 0
        for line in lines:
            if "#" in line:
                x +=1
                image = line[2:-1]
                # print(image)
                data[image] = []
            else:
                line = parse_line(line)
                data[image].append(line)


        # convert it to a pandas df for further preprocessing
        return True, data


    except:
        return False, None
        
    
def process_data(data: dict, dimensions_file_path):

    try:
        
        with open(dimensions_file_path, "r") as f:
                dimensions = json.load(f)

        data_rows = []
        for key in data.keys():
            values = data[key] #list of tuples
            for annot in values:
                row = [key, *annot]
                data_rows.append(row)


        print("dims debug")
        dim_rows = []
        for key in dimensions.keys():
            # print(key)
            # print(dimensions[key])
            # print(type(dimensions[key][0]))
            # break
            row = [key, *dimensions[key]]
            dim_rows.append(row)

            

        # print(lst[:5])

        data_df = pd.DataFrame(data_rows, columns=["id","x", "y", "w", "h", "l1x", "l1y", "l1v", "l2x", "l2y", "l2v", "l3x", "l3y", "l3v", "l4x", "l4y", "l4v", "l5x", "l5y", "l5v"])
        dims_df = pd.DataFrame(dim_rows, columns=["id", "im_w", "im_h"])
        print("data df head")
        print(data_df.head())
        print("dims df head")
        print(dims_df.head())
        full_df = pd.merge(data_df, dims_df, on="id")
        full_df.to_csv("full_df.csv")
        print("full df head:")
        print(full_df.head())
        full_df["x"] = full_df["x"] + full_df["w"]/2
        full_df["y"] = full_df["y"] + full_df["h"]/2
        full_df["x"] = full_df["x"]/full_df["im_w"]    
        full_df["w"] = full_df["w"]/full_df["im_w"]    
        full_df["y"] = full_df["y"]/full_df["im_h"]
        full_df["h"] = full_df["h"]/full_df["im_h"]
        for i in range(1, 6):
            full_df[f"l{i}x"] = (full_df[f"l{i}v"] == True)*(full_df[f"l{i}x"] / full_df["im_w"]) + (~full_df[f"l{i}v"])*(0)
            full_df[f"l{i}y"] = (full_df[f"l{i}v"] == True)*(full_df[f"l{i}y"] / full_df["im_h"]) + (~full_df[f"l{i}v"] == False)*(0)
            # full_df[f"l{i}y"] = full_df[f"l{i}y"] / full_df["im_h"]

        print("full df head:")
        print(full_df.head())

        return True, full_df
    except:
        return False, None

    
def split_data(df: pd.DataFrame, split_ratios):

    

    if not math.isclose(sum(split_ratios), 1.0, abs_tol=0.0001):
            return False, None, None, None

    try: 
        grouping = df.groupby("id")
        print(grouping)
        result = (
        df.groupby("id")
        .apply(lambda group: group.drop(columns=["id", "im_h", "im_w"]).values.tolist())
        .to_dict()
        )
        primary_split_ratio = split_ratios[0]
        secondary_split_ratio = split_ratios[1]/(split_ratios[1]+split_ratios[2]) # test

        all_keys = list(result.keys())
        train_keys, test_val_keys = train_test_split(all_keys, train_size=primary_split_ratio)
        val_keys, test_keys = train_test_split(test_val_keys, train_size=secondary_split_ratio)

        train = {k:result[k] for k in train_keys}
        valid = {k:result[k] for k in val_keys}
        test = {k:result[k] for k in test_keys}


        return True, train, valid, test

    except:
        return False, None, None, None


def write_data(split: dict, path):
    def stringify_label(label):
        s = "0"
        for i in range(len(label)):
            item = label[i]
            if type(item) is float:
                item = round(item, 6)
            elif type(item) is bool:
                item = float(item)

            s+=f" {item}"

        return s


    try:
        keys = list(split.keys())
        for key in keys:
            labels = split[key]
            str_label = ""
            for label in labels:
                str_label+=stringify_label(label)
                str_label+="\n"
            str_label = str_label[:-1]
                # print("LABEL: ", str_label)
            key:str = key[:-4]
            # key = key.replace("/", "_")
            key = key[key.find("/")+1:]
            txt_file_path = os.path.join(path, f"{key}.txt")

            file = open(txt_file_path, mode="w+")
            file.write(str_label)
            file.close()


        return True
    except Exception as e:
        print(e)
        return False


def write_keys(data, path, file_name):

    try:
        keys = list(data.keys())

        OUTPUT_KEYS_FILE_PATH = os.path.join(path, file_name+".json")
        with open(OUTPUT_KEYS_FILE_PATH, "w") as f:
            json.dump(keys, f, indent=4)

        return True
    except:
        return False
    





def main():
    args = sys.argv
    extract = True
    remove_dir = True
    if args[1] == "0":
        extract = False
    if args[2] == "0":
        remove_dir = False


    BASE_DIR = "../data"
    # BASE_RAW_LABELS_DIR = "../data/raw_labels"
    BASE_RAW_LABELS_DIR = os.path.join(BASE_DIR, "raw_labels")
    print("BASE RAW LABELS DIR:", BASE_RAW_LABELS_DIR)
    ZIP_PATH = os.path.join(BASE_RAW_LABELS_DIR, "retinaface_gt_v1.1.zip")
    FILE_NAME = "retinaface_gt_v1.1"
    OUTPUT_DIR = os.path.join(BASE_RAW_LABELS_DIR, FILE_NAME)
    if extract:
        print("extracting data")
        success = extract_data(zip_file_path= ZIP_PATH, output_dir= OUTPUT_DIR)
        check_success_state(success, fail_message="Error while extracting data", pass_message="Data extracted successfully")


    DATA_DST_BASE = os.path.join(BASE_DIR, FILE_NAME)
    if remove_dir:
        print(DATA_DST_BASE)
        if os.path.exists(DATA_DST_BASE):
            print("removing. ...")
            shutil.rmtree(DATA_DST_BASE)
    
    success, images_subdir_path, labels_subdir_path = create_data_env_in_yolo_format(DATA_DST_BASE)
    check_success_state(success, fail_message="Error while creating directories", pass_message="Data environment in yolo format created successfully")


    success, data = obtain_data(extracted_raw_labels_base_dir= OUTPUT_DIR)
    check_success_state(success, fail_message="Error while obtaining data", pass_message="Data obtained successfully")


    DIMENSIONS_FILE_PATH = os.path.join(BASE_DIR, "dimensions.json")
    success, processed_df = process_data(data, DIMENSIONS_FILE_PATH)
    check_success_state(success, fail_message="Error while preprocessing data", pass_message="data preprocessed successfully")


    success, train, valid, test = split_data(processed_df, split_ratios=[0.65, 0.1, 0.25])
    check_success_state(success, fail_message="Error while splitting data", pass_message="data splitted successfully")
    train_keys = list(train.keys())
    val_keys = list(valid.keys())
    test_keys = list(test.keys())
    print("COUNT TRAIN KEYS: ", len(train_keys))
    print("COUNT VALID KEYS: ", len(val_keys))
    print("COUNT TEST KEYS: ", len(test_keys))
    print("COUNT TOTAL KEYS: ", len(train_keys) + len(val_keys) + len(test_keys))


    TRAIN_LAEBLS_PATH = os.path.join(labels_subdir_path, "train")
    VALID_LAEBLS_PATH = os.path.join(labels_subdir_path, "val")
    TEST_LAEBLS_PATH = os.path.join(labels_subdir_path, "test")
    success = write_data(train, TRAIN_LAEBLS_PATH)
    check_success_state(success, "Error while writing data", "data written successfully")
    success = write_data(valid, VALID_LAEBLS_PATH)
    check_success_state(success, "Error while writing data", "data written successfully")
    success = write_data(test, TEST_LAEBLS_PATH)
    check_success_state(success, "Error while writing data", "data written successfully")

    success = write_keys(train, BASE_DIR, "train_keys")
    check_success_state(success, "Error while writing keys", "keys written successfully")
    success = write_keys(valid, BASE_DIR, "valid_keys")
    check_success_state(success, "Error while writing keys", "keys written successfully")
    success = write_keys(test, BASE_DIR, "test_keys")
    check_success_state(success, "Error while writing keys", "keys written successfully")



if __name__ == "__main__":
    main()