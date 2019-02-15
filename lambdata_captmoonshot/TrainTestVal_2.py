import numpy as np
import pandas as pd

def train_test_val_2(df=None):
	import numpy as np
	import pandas as pd
	train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

	"""
	print("training set:\n,", train[:5])
	print()
	print("validation set:\n,", validate[:5])
	print()
	print("test set:\n", test[:5])
	print()
	"""
	len_df = len(df)
	len_train = len(train)
	len_validate = len(validate)
	len_test = len(test)
	"""
	print("Size of dataset along with percent of total DataFrame:\n")
	print("\nLenth of df:\n{}".format(len_df))
	print("\nLenth of training set:\n{}, {}%".format(len_train, round(len_train / len_df, 2) * 100))
	print("\nLenth of validation set:\n{}, {}%".format(len_validate, round(len_validate / len_df, 2) * 100))
	print("\nLenth of test:\n{}, {}%".format(len_test, round(len_test / len_df, 2) * 100))
	"""

	return len_df

	

