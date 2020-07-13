import torch
from model import Scale_ResNet
import pytorch_lightning as pl
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Scale_ResNet()
trainer = pl.Trainer(gpus=8, max_epochs=10, distributed_backend='ddp')
trainer.fit(model)

trainer.test()