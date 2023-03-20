'''script to pad and patch images'''

from tifffile import imread, imwrite
from os import listdir
from patchify import patchify
import numpy as np

#change the paths
imgs_path = '/global/cfs/projectdirs/m636/perciano/Natalie/Prediction/rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5/Original/' 
save_path='/global/cfs/projectdirs/m636/perciano/Natalie/Prediction/rec20160323_093947_232p3_cured_1p5cm_cont_4097im_1500ms_17keV_10.h5/Original_patches/'

files = listdir(imgs_path)
files.sort()

c=0;
for f in files[158:1158]: #set your range
    c=c+1
    image = imread(imgs_path+f)
    image=np.pad(image, pad_width=16) # (2592, 2592)
    patches = patchify(image, (288,288),step=256) # (10, 10, 288, 288)
    filen=f.split('.')[0]
    for i in range(10):
        for j in range(10):
            x=str(i)+str(j)
            imwrite(save_path+filen+'_'+x+'.tif',patches[i][j]) 
                
