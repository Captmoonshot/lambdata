import pandas as pd

def main():
	names = ['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']

	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=names, index_col=False)

	df = df[['age', 'workclass', 'education', 'sex', 'hours-per-week',
             'occupation', 'income']]


	# print(df.head())

	return df






