import os
import fileinput
from tqdm import tqdm
import glob

directory_path = "/ibex/ai/home/toibazd/annotation_extended_10K"
txt_files = glob.glob(os.path.join(directory_path, "*.txt"))


for file_path in tqdm(txt_files):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        modified_text = text.replace(". ", " ")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(modified_text)
print("Replacement and saving complete.")
