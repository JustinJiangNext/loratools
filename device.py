import torch


def fetchDevice(deviceName = "Default", logging = False) -> torch.device:
    device:torch.device = None
    if deviceName == "Default":
        if (torch.backends.mps.is_available()):
            device = torch.device('mps')
            if logging:
                print("Metal Performance Shaders Available! ")
        elif(torch.cuda.is_available()):
            device = torch.device('cuda')
            if logging: 
                print("NVIDIA CUDA Available! ")
        else:
            device = torch.device('cpu')
            if logging:
                print("ONLY CPU Available! ")

    else:
        device = torch.device(deviceName)
    
    return device