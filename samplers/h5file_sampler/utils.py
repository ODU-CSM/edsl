'''
Implementation of utils that handle h5 files

Created on Aug 15, 2021

@author: javon
'''

import h5py as h5

def countExamsInH5s(h5Files):
    '''
    Count the number of examples in given h5 files
    
    Parameter
    ----------
    h5Files : list
        The list of H5 files (tuples, the second element in each tuple is the path).
        It is assumed each file has the sequence dataset, which is 
        a 3D tensor, the 0 dimension represent the example 
    '''
    
    nums = []
    for file in h5Files:
        fileHdl = h5.File(file[1], 'r')
        nums.append(fileHdl['sequence'].shape[0])
    return nums