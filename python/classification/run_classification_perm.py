import pickle as pkl
import sys
from utils.perm_utils import *
from os import path as op
from utils.utils import *
import os

os.environ["OMP_NUM_THREADS"] = "1"   # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"   # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads


if __name__== "__main__":
    
    if len(sys.argv) != 4:
        # no argumemnts, print usage message
        print("Usage:")
        print(" $ python run_classification_perm.py <subID> <input_dir> <out_path>")
        sys.exit(0)
    
    # grab CLI arguments
    subject = sys.argv[1]
    input_dir = sys.argv[2]
    out_path = sys.argv[3]
    perms=1000

    with open(op.join(input_dir, f'{subject}_classification_data.pkl'), 'rb') as f:
        dataset = pkl.load(f)
        
    # specify which analyses to run permutation tests on
    perm_test_dict = {'numerosity_combined': {'format':'combined', 'labels':'labels_numerosity'},
                      'numerosity_dots': {'format':'dots', 'labels':'labels_numerosity'},
                      'numerosity_digits': {'format':'digits', 'labels':'labels_numerosity'}
                     }
                 
    for key in perm_test_dict.keys():
        
        form = perm_test_dict[key]['format']
        labels = perm_test_dict[key]['labels']
        
        perm_results = run_pairwise_uni_classification_perm(dataset, labels, form, perms)
        
        outfile_name = f'{subject}_{labels}_{form}_perm_results.pkl'
        
        with open(op.join(out_path, outfile_name), 'wb') as handle:
            pkl.dump(perm_results, handle, protocol=pkl.HIGHEST_PROTOCOL)
