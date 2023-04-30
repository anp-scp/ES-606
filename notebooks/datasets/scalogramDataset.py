#%%
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
#%%
class CustomBrainMNISTSCalogram(Dataset):
    def __init__(self, eventArrays: np.array, targetsArrays: np.array, extends: tuple) -> None:
        eventTensors = torch.Tensor(eventArrays)
        targets = torch.tensor(targetsArrays)
        self.eventTensors = eventTensors[extends[0]: extends[1]]
        self.targets = targets[extends[0]: extends[1]]
    
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        eeg = self.eventTensors[index]
        target = self.targets[index].item()
        return eeg, target


class CustomBrainMNISTSCalogramAllChannel(Dataset):
    def __init__(self, eventScaloGramDir: str, targetsPath: str, type: str) -> None:
        self.type = type
        self.targets = np.load(targetsPath)
        self.eventScalogramsDir = eventScaloGramDir
        self.currentBatch = 1
        self.eventTensors = None
        if self.type == 'train':
            self.targets = self.targets[:50000]
            self.eventTensors = np.load(eventScaloGramDir + 'saclograms_batch_1.npy')
        elif self.type == 'val':
            self.targets = self.targets[50000:60000]
            self.eventTensors = np.load(eventScaloGramDir + 'saclograms_batch_6.npy')
        else:
            self.targets = self.targets[60000:]
            self.eventTensors = np.load(eventScaloGramDir + 'saclograms_batch_7.npy')
    
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, index):
        if self.type in ['val', 'test']:
            eeg = self.eventTensors[index]
            target = self.targets[index].item()
        else:
            batchToFetch = (index//10000) + 1
            if self.currentBatch != batchToFetch:
                self.eventTensors = None
                self.eventTensors = np.load(self.eventScalogramsDir + f"saclograms_batch_{batchToFetch}.npy")
                self.currentBatch = batchToFetch
            idx = index - ((batchToFetch-1)*10000)
            eeg = self.eventTensors[idx]
            target = self.targets[index].item()
        return eeg, target


class CustomBrainMNISTSLoader():

    def __init__(self, dataset: Dataset, batch_size: int):
        ## our data is already in a random manner. PyTorch Dataloader also works....
        self.dataset = dataset
        self.datasetLength = len(dataset)
        self.batch_size = batch_size
        self.current = 0
    
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        Xs = []
        ys = []
        if self.current >= self.datasetLength:
            raise StopIteration
        for i in range(self.current, self.current + self.batch_size):
            if i >= self.datasetLength:
                break
            x, y = self.dataset[i]
            Xs.append(x)
            ys.append(y)
        Xs = np.stack(Xs, axis=0)
        ys = np.array(ys)
        self.current += self.batch_size
        return torch.from_numpy(Xs), torch.from_numpy(ys)


#%%
if __name__ == "__main__":
    #%%
    dirPath = '../processedData/channelSlectedBandPassed/npy/'
    npData = np.load(dirPath+'eventScalogram.npy')
    npTargets = np.load(dirPath+'targets.npy')
    # %%
    #%%
    randomIndex = np.random.permutation(npTargets.shape[0])
    npData = npData[randomIndex]
    npTargets = npTargets[randomIndex]
    #%%
    trainExtend = (0, 44470)
    validationExtend = (44470, 54470)
    testExtend = (54470, 64470)

    trainDataset = CustomBrainMNISTSCalogram(eventScaloGramDir=npData, targetsArrays=npTargets, extends=trainExtend)
    validationDataset = CustomBrainMNISTSCalogram(eventScaloGramDir=npData, targetsArrays=npTargets, extends=validationExtend)
    testDataset = CustomBrainMNISTSCalogram(eventScaloGramDir=npData, targetsArrays=npTargets, extends=testExtend)