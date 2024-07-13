import json
import jsonlines
import os 
#metadata.jsonl
with open('loraF/metadata.jsonl', 'r') as json_file:
    json_list = list(json_file)

listOfJson = [0]*len(json_list)

counter = 0
for json_str in json_list:
    listOfJson[counter] = json.loads(json_str)
    listOfJson[counter]["text"] = listOfJson[counter]["text"] + ", f4ke"
    counter += 1


with jsonlines.open('metadata2.jsonl', 'w') as writer:
    writer.write_all(listOfJson)
    