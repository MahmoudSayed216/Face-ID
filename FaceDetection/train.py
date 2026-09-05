from ultralytics import YOLO
import yaml
import sys
import os



def main():
    args = sys.argv
    config = args[1]+".yaml"
    BASE_CONFIGS_DIR = "/kaggle/working/Face-ID/FaceDetection/configs"
    TRAIN_CONFIGS_PATH = os.path.join(BASE_CONFIGS_DIR, config)
    COMMON_CONFIGS_PATH = os.path.join(BASE_CONFIGS_DIR, "common_configs.yaml")

    print(f"FINETUNING ON CONFIGS {config}")

    with open(TRAIN_CONFIGS_PATH, "r") as f:
        train_configs = yaml.safe_load(f)
    with open(COMMON_CONFIGS_PATH, "r") as f:
            common_configs = yaml.safe_load(f)
        

    model = YOLO(train_configs["MODEL"])


    results = model.train(
        data=common_configs["DATA_PATH"],
        epochs=train_configs["EPOCHS"],
        imgsz=train_configs["IMAGE_SIZE"],
        batch= train_configs["BATCH_SIZE"],
        workers=train_configs["WORKERS"],
        device=common_configs["DEVICE"],
        project=common_configs["PROJECT"],
        name=train_configs["NAME"],
        exist_ok=False,
        ###
        multi_scale = train_configs["MULTI_SCALE"],
        warmup_epochs = train_configs["WARMUP_EPOCHS"],
        momentum = train_configs["MOMENTUM"],
        lr0 = train_configs["LR0"],
        lrf = train_configs["LRF"],
        freeze = train_configs["FREEZE"],
        cos_lr = train_configs["COS_LR"],
        seed = common_configs["SEED"],
        pretrained = train_configs["PRETRAINED"],
        optimizer = train_configs["OPTIMIZER"]

    )


    print(f"FINETUNING ON CONFIGS {config} completed")







if __name__ == "__main__":
    main()

