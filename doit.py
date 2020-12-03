# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Preprocessing
# %% [markdown]
# ## load data

# %%
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

items = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/items.csv') # supplemental information about the items/products.
salesTrain = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/sales_train.csv') # - the training set. Daily historical data from January 2013 to October 2015.
shops = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/shops.csv') # supplemental information about the shops.
item_categories = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/item_categories.csv') # supplemental information about the items categories.
test = pd.read_csv('C:/Users/jon39/OneDrive/NTNU/3. år/ai/project/test.csv') # the test set. You need to forecast the sales for these shops and products for November 2015.


# %%
# items
cols = items.columns
print("items.csv head")
print(items.head())
print("columns: ", items.columns)
item_names = items[cols[0]]
print("Number of items: ", len(item_names))


# %%
#item categories
cols = item_categories.columns
print("item_categories head")
print(item_categories.head())
cat_names = item_categories[cols[0]]
print("columns: ", cols)
print("number of categories: ", len(cat_names))


# %%
# salestrain

cols = salesTrain.columns
print("Saletrain head")
print(salesTrain.head())
print(salesTrain.describe())
print("columns: ", cols)
print("number of listings: ", len(salesTrain[cols[0]]))
print("max sales: ", np.max(salesTrain[cols[-1]]))
print("min sales: ", np.min(salesTrain[cols[-1]]))
print("shops: ", np.unique(salesTrain[cols[2]]))
print("date_block_num", np.unique(salesTrain[cols[1]]))

unq, coun = np.unique(salesTrain[cols[3]], return_counts=True)
popItem = unq[np.argmax(coun)]
print("item_ids: ", unq)  # not one item per date_block
print("most popular item_id: ", popItem)
print("most popular item name", items['item_name'][items['item_id'] == popItem].values)
corrMatrix = salesTrain.reset_index().corr()
sn.heatmap(corrMatrix, annot=True)




# %%
# test
cols = test.columns
print("test. head")
print(test.head())
print("columns: ", cols)
print("shop ids: ", np.unique(test[cols[1]])) # all shops are to be predicted
print("item ids: ", np.unique(test[cols[2]])) # 3 items not to be predicted

# insert item_price into the dataset and drop nan
avgPrice = salesTrain.groupby('item_id', as_index=False)['item_price'].last()
avgPrice = avgPrice.set_index('item_id')
res = test.join(avgPrice, on='item_id')
testCat = res[res['item_price'].isna()]

res['date_block_num'] = 34

testCat = res[res['item_price'].isna()]
testCat = testCat.drop(columns='item_price')
testCat = testCat.join(items[['item_category_id', 'item_id']].set_index('item_id'))

testCat = testCat.dropna()
res = res.dropna()  # some ids are not in salesTrain
test = res # test contains the ids that have been sold before
print(testCat.head()) # while testCat only contains the ones with category.
print(test.head())

# %%
# plot the sales of most popular item over date_num_block
# may show type of regression to use.

cols = salesTrain.columns
sums = salesTrain[salesTrain[cols[3]] == popItem].groupby(['date_block_num'], as_index=False)['item_cnt_day'].sum() # sum item sales
X1 = sums['date_block_num']
Y1 = sums['item_cnt_day']

sums = salesTrain.groupby(['date_block_num'], as_index=False)['item_cnt_day'].sum() # sum item sales
X2 = sums['date_block_num']
Y2 = sums['item_cnt_day']

X = salesTrain[salesTrain[cols[3]] == popItem]['date_block_num'].values
Y = salesTrain[salesTrain[cols[3]] == popItem]['item_price'].values

X = [5,             7,           10,           11,          12,          15,        20,            25]
Y = [0.354380235, 0.420572942, 0.437437172, 0.440551559, 0.415282362, 0.414365017, 0.397183203, 0.400039768]

X = [4,             8,           16,           32,          40      ]
Y = [0.343980836, 0.482450639, 0.571692306, 0.592347108, 0.591909037]

X = [5,                         10,                 15,           20,          25,        27, 30,     35]
Y = [0.04523255603001597, 0.19970525007751072, 0.382924067919018, 0.5845137681421553, 0.6131795432799545, 0.614663809774078, 0.5993007951645425, 0.6018195690139645]


# %%
# making dataset that is usable, sum over blocknum, merge with category
data = salesTrain.groupby(['date_block_num', 'shop_id', 'item_id', 'item_price'], as_index=False)['item_cnt_day'].sum() # sum item sales
data = data.set_index('item_id').join(items[['item_category_id', 'item_id']].set_index('item_id'))
data = data.reset_index()


# %%
# X=date_block_num, shop_id, item_id Y=item_cnt_day
# could also include price
from sklearn.model_selection import train_test_split

X = data[['date_block_num', 'shop_id', 'item_id']].values
X2 = data[['date_block_num', 'shop_id', 'item_id', 'item_category_id']].values
X3 = data[['date_block_num', 'shop_id', 'item_id', 'item_category_id', 'item_price']].values
X4 = data[['date_block_num', 'shop_id', 'item_id', 'item_price']].values
Y = data['item_cnt_day'].values
Y = Y.reshape(-1,1)
print(X, X2, X3)
print(Y)
print(len(Y))

X_train, X_test, y_train, y_test = train_test_split(X4, Y, test_size=0.3, random_state=0)


# %%


# %%




# %%



# %%
#ensemble of the best models

from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
estimators=[('ExTree', ExtraTreesRegressor(n_estimators=40, max_depth=25, n_jobs=8)), ('RF', RandomForestRegressor(max_depth=32, random_state=0, n_jobs=8))]
ensemble = VotingRegressor(estimators)
#ensemble.fit(X_train, y_train)
#print("trained. started fitting")
#print(ensemble.score(X_test, y_test))


# %%
#X2 = data[['date_block_num', 'shop_id', 'item_id', 'item_category_id']].values

print(test.columns)
toPred = test[['date_block_num', 'shop_id', 'item_id', 'item_price']].values
toPredCat = testCat[['date_block_num', 'shop_id', 'item_id', 'item_category_id']].values
print(toPred)

ensemble.fit(X4, Y)
test['item_cnt_month'] = ensemble.predict(toPred)
ensemble = VotingRegressor(estimators)
ensemble.fit(X2, Y)
testCat['item_cnt_month'] = ensemble.predict(toPredCat)

test = test.append(testCat)
test = test[['ID', 'item_cnt_month']]
test = test.set_index('ID').sort_index()
test = test.reset_index()
test.to_csv(r'submission.csv', index = False, header=True)

