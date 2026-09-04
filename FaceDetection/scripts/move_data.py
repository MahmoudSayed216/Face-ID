import json
import os

def check_success_state(success, fail_message, pass_message):
    if not success:
        print(fail_message)
        exit(1)
    else:
        print(pass_message)
    

def load_keys_json(path, filename):
    json_file_path = os.path.join(path, filename)
    try:
        with open(json_file_path, "r") as f:
            my_list = json.load(f)
        return True, my_list
    except Exception as e:
        print(e)
        return False, None

def move_data(src_dir: str, dst_dir: str, keys: list, split: str):
    print("SOF")
    try:
        dst_dir = os.path.join(dst_dir, split)
        print(dst_dir)
        for key in keys:
            img_dir = os.path.join(src_dir, key)
            print(img_dir)
            break



        return True
    except Exception as e:
        print(e)
        print("_____")
        return False

    
def main():

    SRC_DIR = "/kaggle/input/datasets/iamprateek/wider-face-a-face-detection-dataset/WIDER_train/WIDER_train/images"
    KEYS_DIR = "/kaggle/working/Face-ID/FaceDetection/data"
    DST_DIR = "/kaggle/working/Face-ID/FaceDetection/data/retinaface_gt_v1.1/retinaface/images"


    success, train_keys = load_keys_json(KEYS_DIR, "train_keys.json")
    check_success_state(success, "Error while loading keys file", "Keys file loaded successfully")
    success, test_keys = load_keys_json(KEYS_DIR, "test_keys.json")
    check_success_state(success, "Error while loading keys file", "Keys file loaded successfully")
    success, valid_keys = load_keys_json(KEYS_DIR, "valid_keys.json")
    check_success_state(success, "Error while loading keys file", "Keys file loaded successfully")


    success = move_data(SRC_DIR, DST_DIR, train_keys, "train")
    check_success_state(success, "Error while moving data", "data moved successfully")
    success = move_data(SRC_DIR, DST_DIR, train_keys, "test")
    check_success_state(success, "Error while moving data", "data moved successfully")
    success = move_data(SRC_DIR, DST_DIR, train_keys, "val")
    check_success_state(success, "Error while moving data", "data moved successfully")




if __name__ == "__main__":
    main()
