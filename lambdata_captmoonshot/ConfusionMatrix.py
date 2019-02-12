	
def main():
	
	import pandas as pd
	import numpy as np

	names = ['age', 
	         'workclass', 
	         'fnlwgt', 
	         'education', 
	         'education-num', 
	         'marital-status', 
	         'occupation', 
	         'relationship', 
	         'race', 
	         'sex', 
	         'capital-gain', 
	         'capital-loss', 
	         'hours-per-week', 
	         'native-country', 
	         'income']

	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=names)

	df = df[['age', 'workclass', 'education', 'sex', 'hours-per-week', 'occupation', 'income']]

	df_dummies = pd.get_dummies(df)

	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix

	X = df_dummies.loc[:, 'age':'occupation_ Transport-moving'].values
	y = df_dummies['income_ >50K'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	lr = LogisticRegression().fit(X_train, y_train)

	y_pred = lr.predict(X_test)

	print("\nResults on the UCI Repository Adult Training Dataset:")
	print()
	print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
	print()



	confusion = confusion_matrix(y_test, y_pred)

	print("Confusion matrix:\n{}".format(confusion))
	print()
	true_negative, false_positive, false_negative, true_positive = confusion.ravel()
	print()
	print('True Negative: {}'.format(true_negative))
	print()
	print('False Positive: {}'.format(false_positive))
	print()
	print('False Negative: {}'.format(false_negative))
	print()
	print('True Positive: {}'.format(true_positive))


main()




