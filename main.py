from FaceIDSystem import FaceIDSystem


try:
    system = FaceIDSystem("auto", "YOLO26NANO_BASE", "r18")
    system.run()

except Exception as e:
    print("There's been an error")
    print(e)
    exit(1)