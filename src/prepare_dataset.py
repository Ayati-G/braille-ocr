import os
import shutil #file operations such as copy src,dest & move 
import random 
from pathlib import Path #work with filepaths

SOURCE_DIR = Path(r"C:\Users\Ayati Gupta\Desktop\braille-ocr\dataset")
OUTPUT_DIR = Path(r"C:\Users\Ayati Gupta\Desktop\braille-ocr\data\processed")
TRAIN_SPLIT = 0.8

letters = [chr(i) for i in range(ord('a'),ord('z')+1)] #range not inclusive of last

def prepare():
    for letter in letters:
        all_images =[]
        letter_dir= SOURCE_DIR / letter / "Uploaded"

        for set_folder in letter_dir.iterdir():
            #iterdir() loops over every dir
            for img in set_folder.glob("*.jpg"): # Path.glob("*.jpg") finds all jpg files in a folder.
                all_images.append(img)
        
        random.shuffle(all_images)
        split_index = int(len(all_images)*TRAIN_SPLIT)
        train_imgs= all_images[:split_index]
        val_imgs = all_images[split_index:]

        output_train = OUTPUT_DIR / "train" / letter
        output_train.mkdir(parents=True, exist_ok=True) #create folder if dne, create missing parent & no crashing if folder exists
        for imgs in train_imgs:
            shutil.copy(imgs, output_train)
        
        output_val = OUTPUT_DIR / "val" / letter
        output_val.mkdir(parents=True, exist_ok=True)
        for imgs in val_imgs:
            shutil.copy(imgs, output_val)

if __name__ == "__main__":
    prepare()
    print("Mission Accomplished! <3")