import torch
import os
from multiprocessing import Process, Manager
from PIL import Image
import fnmatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from math import log as ln
import shutil

import torchvision.transforms as transforms

from CIFAKEClassifier import CIFAKEClassifier
from device import fetchDevice


import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def inv_sigmoid(x):
    return ln(x) - ln(1 - x)
     

def loadSubFolder(path, result, id, device = fetchDevice()) -> torch.tensor:
    imageTensors:list[torch.tensor] = []
    allImages = sorted(os.listdir(path), key=numericalSort)
    for image in allImages:  
        if fnmatch.fnmatch(image, f"*.jpg") or fnmatch.fnmatch(image, f"*.png"): 
            im = Image.open(path + "/" + image)
            im = im.resize((32, 32))
            imageTensors.append(transforms.ToTensor()(im))

    result[id] = torch.stack(imageTensors)



def loadAllData(folder, device = fetchDevice()) -> torch.tensor:
    process:list[Process] = []

    with Manager() as manager:
        subFolderTensors = manager.list()
        subFolderTensors.extend([0]*len(os.listdir(f"{folder}")))

        f_id = 0
        for subFolder in os.listdir(f"{folder}"):
            if os.path.isdir(f"{folder}/{subFolder}"):
                process.append(Process(target=loadSubFolder, args = (f"{folder}/{subFolder}", subFolderTensors, f_id, device)))
                f_id += 1
                if(f_id >= 1):
                    break


        for p in process:
            p.start()
        for p in process:
            p.join()
        
        #print(subFolderTensors)
        return torch.cat(tuple(subFolderTensors)).to(device)
        



def loadOneFolder(path, device = fetchDevice()) -> torch.tensor:
    imageTensors:list[torch.tensor] = []
    allImages = sorted(os.listdir(path), key=numericalSort)
    for image in allImages:  
        if fnmatch.fnmatch(image, f"*.jpg") or fnmatch.fnmatch(image, f"*.png"): 
            im = Image.open(path + "/" + image)
            im = im.resize((32, 32))
            imageTensors.append(transforms.ToTensor()(im))

    return torch.stack(imageTensors).to(device)

def getFakest(ratingList):
    ratingMap = {ratingList[i]: i for i in range(0, len(ratingList))}
    sortedMap = collections.OrderedDict(sorted(ratingMap.items(), reverse=False))
    res = []

    for k, v in sortedMap.items():
        res.append(v)
        if(len(res) >= 600):
            break
        
    return res

def filter(filterList, path, prefix):
    #image_2914.png
    for i in filterList:
        shutil.copy(f"data/{path}/image_{i}.png", f"loraF/{prefix}_image_{i}.png")


if __name__ == '__main__':
    model = CIFAKEClassifier()
    model.to(fetchDevice())
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    results = model(loadOneFolder("data2/sd_2_1_lora_dogs_with_modifiers_no_trigger_negative_3")).squeeze().tolist()

    total = 0.0
    for res in results:
        total += res

    print(float(total)/float(len(results)))

    """
    filter(getFakest(model(loadOneFolder("data/SD21Airplane")).squeeze().tolist()), "SD21Airplane", "airplane")
    filter(getFakest(model(loadOneFolder("data/SD21Automobile")).squeeze().tolist()), "SD21Automobile", "automobile")
    filter(getFakest(model(loadOneFolder("data/SD21Bird")).squeeze().tolist()), "SD21Bird", "bird")
    filter(getFakest(model(loadOneFolder("data/SD21Cat")).squeeze().tolist()), "SD21Cat", "cat")
    filter(getFakest(model(loadOneFolder("data/SD21Deer")).squeeze().tolist()), "SD21Deer", "deer")
    filter(getFakest(model(loadOneFolder("data/SD21Dog")).squeeze().tolist()), "SD21Dog", "dog")
    filter(getFakest(model(loadOneFolder("data/SD21Frog")).squeeze().tolist()), "SD21Frog", "frog")
    filter(getFakest(model(loadOneFolder("data/SD21Horse")).squeeze().tolist()), "SD21Horse", "horse")
    filter(getFakest(model(loadOneFolder("data/SD21Ship")).squeeze().tolist()), "SD21Ship", "ship")
    filter(getFakest(model(loadOneFolder("data/SD21Truck")).squeeze().tolist()), "SD21Truck", "truck")
"""



    
