'''script to split and move data to respective train/val/test folders'''
from pathlib import Path
import shutil
import os
 
# defining source and destination
# paths
src = '/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/train/img'
trg = '/global/cfs/projectdirs/m636/AJ/p1_models/VTFS/data/splits'
textfilepath = '/global/cfs/projectdirs/m636/AJ/p1_models/VTFS/data/splits/val_10.txt'

images=[line.strip() for line in open(textfilepath)]

for elem in images:
    elem=elem+'.tif'
    shutil.copy2(os.path.join(src,elem), trg)
 


