import torch
import theseus as th

import theseus.utils.examples as theg
from theseus.utils.examples.bundle_adjustment.data import *

torch.manual_seed(1)

# Smaller values result in error
th.SO3.SO3_EPS = 1e-6

if True:
    ba = BundleAdjustmentDataset.generate_synthetic(30, 1000)
    print("\nBA:")
    ba.histogram()

    path = "/tmp/test.txt"
    ba.save_to_file(path)

    ba2 = BundleAdjustmentDataset.load_from_file(path)
    print("\nBA2:")
    ba2.histogram()

ba3 = BundleAdjustmentDataset.load_from_file("/home/maurimo/BAL/problem-49-7776-pre.txt")
print("\nBA3:")
ba3.histogram()