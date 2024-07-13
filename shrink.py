import os
from PIL import Image 
from pathlib import Path
from multiprocessing import Process



def pipetoPNGJPG(directory, fileName, outDirectoryPNG, outDirectoryJPG):
    im = Image.open(directory + "/" + fileName)
    im = im.resize((32, 32))
    im.save(outDirectoryPNG + "/" + Path(fileName).stem + ".png")
    im.save(outDirectoryJPG + "/" + Path(fileName).stem + ".jpg")

#pipetoPNGJPG("myData/cars", "testcar.png", "output")
    

def convertFolder(directory, outDirectoryPNG, outDirectoryJPG):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            pipetoPNGJPG(directory, filename, outDirectoryPNG, outDirectoryJPG)
        else:
            print("WARNING: NON PNG IMAGE DETECTED")
    print("Finished converting " + directory)

def convertWrapper(imageFolder, subject, outDirectory):
    outDirectoryJPG = outDirectory + "JPG"
    outDirectoryPNG = outDirectory + "PNG"
    try:
        os.system("mkdir " + outDirectoryJPG)
        os.system("mkdir " + outDirectoryPNG)
    except:
        pass
    
    outDirectoryJPGSubject = outDirectoryJPG + "/" + subject
    outDirectoryPNGSubject = outDirectoryPNG + "/" + subject
    os.system("mkdir " + outDirectoryJPGSubject)
    os.system("mkdir " + outDirectoryPNGSubject)
    print(outDirectoryPNGSubject)

    convertFolder(imageFolder + "/" + subject, outDirectoryPNGSubject, outDirectoryJPGSubject)

if __name__ == '__main__':
    print("Main start")
    dataFolderName = input("Data Folder name ")
    converters = [
        Process(target=convertWrapper, args=(dataFolderName, "SD21Airplane", dataFolderName + "output")),
        Process(target=convertWrapper, args=(dataFolderName, "SD21Automobile", dataFolderName + "output")),
        Process(target=convertWrapper, args=(dataFolderName, "SD21Bird", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Cat", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Deer", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Dog", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Frog", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Horse", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Ship", dataFolderName + "output")),
    	Process(target=convertWrapper, args=(dataFolderName, "SD21Truck", dataFolderName + "output")),

    ]

    for worker in converters:
        worker.start()
    for worker in converters:
        worker.join()
    