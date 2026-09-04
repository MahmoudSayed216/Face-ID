FILE1_PATH = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/experminets/retinaface_gt_v1.1/train/label.txt"
FILE2_PATH = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/experminets/retinaface_gt_v1.1/test/label.txt"
FILE3_PATH = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/experminets/retinaface_gt_v1.1/val/label.txt"

file1 = open(FILE1_PATH)
file2 = open(FILE2_PATH)
file3 = open(FILE3_PATH)


f1_lines = file1.readlines()
f2_lines = file2.readlines()
f3_lines = file3.readlines()

f1_faces = [line for line in f1_lines if "#" not in line]
f1_lines = [line for line in f1_lines if "#" in line]
f2_lines = [line for line in f2_lines if "#" in line]
f3_lines = [line for line in f3_lines if "#" in line]
print("train lines: ",  len(f1_lines))
print("test lines: ",   len(f2_lines))
print("val lines: ",    len(f3_lines))
print("Total: ", len(f1_lines)+len(f2_lines)+len(f3_lines))
print("Number of faces in train: ", len(f1_faces))

file1.close()
file2.close()
file3.close()
