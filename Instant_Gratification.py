## This is just for understanding
## Instant Gratification kaggle challenge
import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Step 1 and 2 - Build first QDA model and predict test

cols = [c for c in train.columns if c not in ['id', 'traget']]

cols.remove('wheezy-copper-turtle-magic')

oof = np.zeros(len(train))
preds = np.zeros(len(test))

# Build 512 separate models
# for i in range(4):  # add iteration & scaling variance threshold gives 97% accuracy

for i in range(512):

	# ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS i
	# wheezy column has values range 0 - 512 in 13000 rows.[i.e, 512 unique vals]
	# train['wheezy-copper-turtle-magic'] == i shows true or false the statement is for all rows in that col
	# train[train['wheezy-copper-turtle-magic'] == i] shows only the true val rows and all cols
	train2 = train[train['wheezy-copper-turtle-magic'] == i]  # print(train2) -> [502 rows x 258 columns]
	test2 = test[test['wheezy-copper-turtle-magic'] == i]

	idx1 = train2.index
	idx2 = test2.index

	train2.reset_index(drop=True, inplace=True) # index now becomes a new col and only this col remain and every col is drops-> reset_index(drop=True)
						    # train2 data remains same after train2.reset_index(drop=True, inplace=True)
	# FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

	sel = VarianceThreshold(threshold = 1.5) # can also use -> pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
	sel.fit(train2[cols])                                   #  pipe.fit_transform(train2[cols])

	train3 = sel.transform(train2[cols])
	test3 = sel.transform(test2[cols])

	# STRATIFIED K-FOLD

	skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True) # target -> is the label col 0 or 1. 

	for train_index, test_index in skf.split(train3, train2['target']): 
		# 11 times split of X_train and y_train(think like this)

		# Model and predict with QDA

		clf = QuadraticDiscriminantAnalysis(reg_param = 0.5)
		clf.fit(train3[train_index, :], train2.loc[train_index]['target'])

		oof[idx1[test_index]] = clf.predict_proba(train[test_index, :]) [:, 1]

		preds[idx2] += clf.predict_proba(test3) [:, 1] / skf.n_splits


auc = roc_auc_score(train['traget'], oof)
print('QDA scores CV =', round(auc, 5))

# Step 3 & 4 - Add pseudo label data and build second model

test['target'] = preds

oof = np.zeros(len(train))
preds = np.zeros(len(test))

# Build 512 separate models

for k in range(512):
	# ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS i
	train2 = train[train['wheezy-copper-turtle-magic'] == k]
	train2p = train2.copy()
	idx1 = train2.index

	test2 = test[test['wheezy-copper-turtle-magic'] == i]
	idx2 = test2.index

	# Add pseudo labeled data

	test2p = test2[(test2['target'] <= 0.1)  | (test2['target'] >= 0.99)].copy()

	test2p.loc[test2p['target'] >= 0.5, 'target'] = 1
	test2p.loc[test2p['target'] < 0.5, 'target'] = 0

	train2p = pd.concat([train2p, test2p], axis=0)

	train2p.reset_index(drop=True, inplace=True)

	# FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

	sel = VarianceThreshold(1.5)  # we can also use PCA here
	sel.fit(train2p[cols])

	train3p = sel.transform(train2p[cols])
	train3 = sel.transform(train2[cols])
	test3 = sel.transform(test2[cols])

	# STRATIFIED K FOLD

	skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

	for train_index, test_index in skf.split(train3p, train2p['target']):

		test_index3 = test_index[test_index < len(train3)]  # all true rows are taken out

		clf = QuadraticDiscriminantAnalysis(reg_param = 0.5)
		clf.fit(train3p[train_index, :], train2p.loc[train_index]['target'])

		oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3, :]) [:, 1]

		preds[idx2] += clf.predict_proba(test3) [:, 1] / skf.n_splits

auc = roc_auc_score(train['target'], oof)
print('Pseudo labeled QDA scores CV =', round(auc, 5))

# This is for submission of the roc_auc_score

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)

















 
