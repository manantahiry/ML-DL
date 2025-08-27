
import os
import json
import pandas as pd

#read the ods file
data = pd.read_excel("datasetDIALECT.ods")

#function to create a dictionary from the words and their paths
def create_dict(words, base_path):
    words_json = []
    for index, word in enumerate(words):
        word = word + ".wav"
        path = os.path.join(base_path, word)
        if os.path.exists(path):
            print(f"File found: {path}")
            mot = {
                "id": index,
                "dialect": word,
                "path": path,
            }
            words_json.append(mot)
    return words_json

#function to create a json file from the data
def json_data(data, words, base_path, file_name):
    dialects = data[words]
    result = create_dict(words=dialects, base_path=base_path)

    # Create the JSON structure
    json_data_ = json.dumps(result, indent=4)

    # Save the JSON data to a file
    with open(f"{file_name}.json", "w") as json_file:
        json_file.write(json_data_)

    # Print the JSON data
    print(f"Content of {file_name}.json:")
    print(json_data)

# Create JSON files for each dialect and official word
words_list = ["DIALECT", "OFFICIEL"]
base_paths = ["/audio_dialect", "/audio_officiel"]

# Iterate over the words and base paths to create JSON files
for base_path, words in zip(base_paths, words_list):
    json_data(data=data, words=words, base_path=base_path, file_name=words)

