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

# %% [markdown]
# ### items
# 

# %%
cols = items.columns
print("items.csv head")
print(items.head())
print("columns: ", items.columns)
item_names = items[cols[0]]
print("Number of items: ", len(item_names))

# %% [markdown]
# ### item_categories

# %%
cols = item_categories.columns
print("item_categories head")
print(item_categories.head())
cat_names = item_categories[cols[0]]
print("columns: ", cols)
print("number of categories: ", len(cat_names))

# %% [markdown]
# ### test

# %%
cols = test.columns
print("test. head")
print(test.head())
print("columns: ", cols)
print("shop ids: ", np.unique(test[cols[1]])) # all shops are to be predicted
print("item ids: ", np.unique(test[cols[2]])) # 3 items not to be predicted

# %% [markdown]
# ### salesTrain
# 
# predicting sum of sales of each item in each store. 

# %%
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
corrMatrix = salesTrain.reset_index().corr()
sn.heatmap(corrMatrix, annot=True)
plt.title("Correlation matrix of salesTrain")
plt.show()


# %%
# plot the sales of most popular item over date_num_block
# may show type of regression to use.
X = salesTrain[salesTrain[cols[3]] == popItem][cols[1]]
Y = salesTrain[salesTrain[cols[3]] == popItem][cols[-1]]
plt.plot(X, Y)
plt.show()
print(cols[-1])


# %%
# making dataset that is usable, sum over blocknum, merge with category
data = salesTrain.groupby(['date_block_num', 'shop_id', 'item_id', 'item_price'], as_index=False)['item_cnt_day'].sum() # sum item sales
data = data.set_index('item_id').join(items[['item_category_id', 'item_id']].set_index('item_id'))
data = data.reset_index()

corrMatrix = data.corr()
sn.heatmap(corrMatrix, annot=True)
plt.title("Correlation matrix of salesTrain")
plt.show()


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

# %% [markdown]
# ### linear regression

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=False)
model.fit(X_train, y_train)
print(model.score(X_test,y_test))


# %%
# SVR, support vector regression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

regr = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=4))
regr.fit(X_train, y_train)


# %%
print(regr.score(X_test, y_test))


