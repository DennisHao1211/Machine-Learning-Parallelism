import torch

# pytorch allows use to easily use different hardware without changing your model code
# it has built in methods to detect whethere we are on a cude machine (gpu) or a mac with an apple gpu (mps)
# if none of those exist running on a cpu will always work
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print("Detected device:", device)
