import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# # display graphs in notebook
# %matplotlib inline 

# read csv file as a dataframe
df = pd.read_csv('911.csv')

# check features of dataframe
df.head()
df.info()
df.describe()

# top 5 zipcodes for 911 calls
df['zip'].value_counts().head()

# top 5 townships (twp) for 911 calls
df['twp'].value_counts().head()

# number of unique codes
df['title'].nunique()

# create new column with reason for call, extracting reason from strings in column 'title'
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

# countplot of 911 calls by Reason
sns.countplot(x='Reason', data=df)

# convert column with timestamps from strings to DateTime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# create columns with hour, month, and day of week Hour, Month, and Day of Week
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.day_name())

# create countplots for Day of Week and Month columns with Reason column as hue
sns.countplot(x='Day of Week', data=df, hue='Reason')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.countplot(x='Month', data=df, hue='Reason')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# dataset missing some months - create a linear fit on the number of calls per month
byMonth = df.groupby('Month').count()
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())

# create a new column that contains date from the timeStamp column and create a plot of counts of 911 calls by date
df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['lat'].plot()
plt.tight_layout()

# create a plot of counts by date for each reason for 911 call ('lat' column used to count total entries in dataframe)
df[df['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()

df[df['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()

df[df['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()

# restructure dataframe and create heatmap and clustermap by Day of Week and Hour
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
