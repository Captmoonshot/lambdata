import numpy as np
import pandas as pd

def train_test_val(df=None):
	import numpy as np
	import pandas as pd
	train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
	
	print("training set:\n,", train[:5])
	print()
	print("validation set:\n,", validate[:5])
	print()
	print("test set:\n", test[:5])