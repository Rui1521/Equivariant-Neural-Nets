#
# Set up profiling -- for pytorch lightning this needs to be done first
#

import profiler
profiler.enable() # run profiler.disable() to turn off profiling

if profiler.TIMEMORY_AVAILABLE and profiler.Profiler().enabled:
    import timemory

    # configure how timemory reports data
    timemory.settings.flat_profile = True
    timemory.settings.timeline_profile = False



#
# Training
#

import torch
from model import Scale_ResNet
import pytorch_lightning as pl
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Scale_ResNet()
trainer = pl.Trainer(gpus=1, max_epochs=1, distributed_backend='dp')
trainer.fit(model)

trainer.test()



#
# Finalize profiling
#

if profiler.TIMEMORY_AVAILABLE and profiler.Profiler().enabled:
    timemory.finalize()
