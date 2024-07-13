
"""

from transformers import pipeline

from PIL import Image 
  
# open method used to open different extension image file 
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0, max_new_tokens = 20)


#captioner("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png")
#print(help(captioner))



im = Image.open(r"DOG.png")  

#cache = captioner("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png")
cache = captioner(im)

#cache[0]['generated_text'] = "hello"
print(cache)


## [{'generated_text': 'two birds are standing next to each other '}]


"""
"""
import os
for x in os.listdir("loraF"):
    print(x)
"""



from transformers import pipeline
from PIL import Image 
import jsonlines
import os 


# open method used to open different extension image file 
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0, max_new_tokens = 20)


totalImgs = len(os.listdir("loraF"))
metaList = [0]*totalImgs

imgNum = 0
for file in os.listdir("loraF"):

    textDescription = captioner(Image.open(f"loraF/{file}"))[0]["generated_text"]
    metaList[imgNum] = {"file_name": file, "text": textDescription}
    
    imgNum += 1
    print(imgNum)

with jsonlines.open('metadata.jsonl', 'w') as writer:
    writer.write_all(metaList)
    