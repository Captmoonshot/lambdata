import unittest
from TrainTestVal_2 import *
import pandas as pd
from load_adult_data import *

data = main()

"""
### The following lines if for use if we want to use load_adult_data instead of manually ###
### loading pandas DataFrames ###

names = ['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=names, index_col=False)

df = df[['age', 'workclass', 'education', 'sex', 'hours-per-week',
             'occupation', 'income']]
"""


class TrainTestValTests(unittest.TestCase):
	""" Tests the train_test_val function which returns the length
	of the dataset, in this case the adult dataset from UCI which is
	imported at the top and then instantiated with variable data.
	"""


	def test_sizes(self):
		self.assertEqual(train_test_val_2(data), len(data))

	#def test_sizes_2(self):


if __name__ == '__main__':
	unittest.main()





