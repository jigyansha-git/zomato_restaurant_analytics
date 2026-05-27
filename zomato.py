import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns

df = pd.read_csv(r"D:\kaggle_Dataset\zomato.csv")
df.head()
df.isnull().sum()

df['rate'] = df['rate'].str.replace('/5','')
df['rate'] = df['rate'].replace(['NEW','-'],np.nan)
df['rate'] = df['rate'].astype(float)

df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',','')
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)

df['dish_liked'] = df['dish_liked'].fillna('Not Mentioned')

df.drop('phone',axis=1,inplace=True)

df['location'].fillna(df['location'].mode()[0],inplace=True)
df['rest_type'].fillna(df['rest_type'].mode()[0],inplace=True)
df['cuisines'].fillna(df['cuisines'].mode()[0],inplace=True)

df['rate_filled'] = df['rate'].fillna(df['rate'].median())

df.duplicated().sum()

df.describe()

df['location'].value_counts().head(10)
df.groupby('location')['rate_filled'].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.histplot(df['rate_filled'],bins=20)
plt.title('Distribution of rating')
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(y=df['location'],order=df['location'].value_counts().head(10).index)
plt.title('Top restraunt locations')
plt.show()

df['online_order'].value_counts()

sns.boxplot(x='online_order',y='rate_filled',data=df)
plt.title("Online order vs Ratings")
plt.show() #Restaurants with online ordering tend to maintain slightly higher ratings.

sns.boxplot(x='book_table',y='rate_filled',data=df)
plt.title("Table booking vs Ratings")
plt.show()  # Restaurants offering table booking generally have higher customer ratings

plt.figure(figsize=(8,5))
sns.histplot(df['approx_cost(for two people)'],bins=30)
plt.title('Cost Distribution')
plt.show()

sns.scatterplot( x='approx_cost(for two people)', y='rate_filled',data=df)
plt.title('Cost vs Rating')
plt.show()  #Higher restaurant pricing does not guarantee higher customer ratings.

df['cuisines'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.countplot(y=df['cuisines'],order=df['cuisines'].value_counts().head(10).index)
plt.title('Top cuisines')
plt.show()


df['rest_type'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.countplot(y=df['rest_type'],order=df['rest_type'].value_counts().head(10).index)
plt.title('Restraunt Types')
plt.show()

top_votes = df.sort_values(by='votes',ascending=False)
top_votes[['name','votes','rate_filled']].head(10)

sns.scatterplot(x='votes',y='rate_filled',data=df)
plt.title('Votes vs Rating')
plt.show()  #Highly-rated restaurants with low votes may lack visibility rather than quality

#Numerical Correlation
numeric_df = df.select_dtypes(include=['float64','int64'])
corr = numeric_df.corr()

sns.heatmap(corr,annot=True)
plt.title('Correlation heatmap')
plt.show()

# Feature engineering
def cost_category(x):
    if x <= 500:
        return 'Budget'
    elif x <= 1500:
        return 'Mid-range'
    else:
        return 'Premium'
df['cost_category'] = df['approx_cost(for two people)'].apply(cost_category)


def rating_category(x):
    if x >= 4:
        return 'Excellent'
    elif x >= 3:
        return 'Good'
    else:
        return 'Average'

df['rating_category'] = df['rate_filled'].apply(rating_category)


# top 10 highest rated restaurant
top_rated = df.sort_values(by='rate_filled',ascending=False)
top_rated[['name','rate_filled']].head(10)

# Most expensive restaurant
expensive = df.sort_values(by='approx_cost(for two people)',ascending=False)
expensive[['name','approx_cost(for two people)']].head(10)


columns_to_drop = [
    'url',
    'address',
    'dish_liked',
    'reviews_list',
    'menu_item'
]

df.drop(columns=columns_to_drop, inplace=True)


df.to_csv("Analyzed_zomato_data.csv",index=False)


























































