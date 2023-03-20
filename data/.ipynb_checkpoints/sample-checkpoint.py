'''script to split data into train val test ; input:paths to data  output: text files'''


import mmcv
import os
import os.path as osp
import random 

#define variables
global data_root, img_dir, ann_dir, classes, split_dir
data_root = '/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/train' #the parent directory of image and mask folders
img_dir = '/img' #subpath
classes = ('background', 'bead') 

global train,val,test
train,val,test = 50000,10000,0 #set your train:val split


def datasplit(data_root,img_dir,ann_dir,train,val,test): 
    '''
    allots file names into train val test lists
    '''    
    ground=[]
    data=[]
    training, validation, testing = [], [], []

    for name in os.listdir(data_root+img_dir):
        ground.append(name)

    for i in range(train+val+test):
        data+=[ground[i]] #entire data
        
    training = random.sample(data, train)
    data1 = [x for x in data if x not in training] #data-train (to avoid overlap)
    validation = random.sample(data1, val)
    data2 = [x for x in data1 if x not in validation] #data1-val (to avoid overlap)
    testing = random.sample(data2, test)
    
    return(training, validation,testing) #lists


def splittxtfile(data_root, train, val, test):
    '''
    writes the filenames into a textfile
    '''
    
    tr,v,te = datasplit(data_root,img_dir,ann_dir,train,val,test)
    split_dir = './splits' #saves all the text files to this folder
    print(tr)
    mmcv.mkdir_or_exist(osp.join(split_dir))
    with open(osp.join(split_dir, 'train_'+str(train)+'.txt'), 'w') as f:
      f.writelines(osp.splitext(elem)[0] + '\n' for elem in tr)
    with open(osp.join(split_dir, 'val_'+str(val)+'.txt'), 'w') as f:
      f.writelines(osp.splitext(elem)[0] + '\n' for elem in v)
    with open(osp.join(split_dir, 'test_'+str(test)+'.txt'), 'w') as f:
      f.writelines(osp.splitext(elem)[0] + '\n' for elem in te)

def main():
    splittxtfile(data_root, train, val, test) 

if __name__ == "__main__":
    main()    
