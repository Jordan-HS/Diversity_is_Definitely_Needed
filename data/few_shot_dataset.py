class FewShotDataset:
    
    def __init__(self, data, transforms=None) -> None:
        self.transforms = transforms
        _data = []
        
        for cls in data:
            for imgs in data[cls]:
                _data.append((cls, imgs))
                
        self.data  = _data
        
    def __getitem__(self, index):
        label, img = self.data[index]
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, label
        
    def __len__(self):
        return(len(self.data))