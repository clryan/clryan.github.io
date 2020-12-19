---
layout: post
title: "Customer Spending Prediction 1: Processing Large Data Files in Pandas"
description: "Google Analytics offers a ton of information about website traffic. In this post, we'll look at using that data to predict customer spending at the Google Store."
is_post: true
tags: [python, data cleaning]
---

[This Kaggle competition](https://www.kaggle.com/c/ga-customer-revenue-prediction) challenges participants to analyze Google Analytics data from the Google Merchandise Store (aka G Store, where Google products are sold) to predict future revenue by site visitor. There's just one problem: the data files are a whopping 34 GB - ouch. Dealing with very large data files is a problem that most data scientists will run into eventually. In this post, I'll walk through how I loaded and processed this giant data set without tearing my hair out.

In this notebook, we'll load in the data, examine it, fix any data types and drop useless columns, and save our processed data as a feather file for faster loading.

```python
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
from pandas.io.json import json_normalize
import gc
import psutil
import os
%matplotlib inline

# helper functions for data cleaning
!pip install git+http://github.com/clryan/pyutils.git
from pyutils.cleaning import na_cols, one_val_cols
```

### Read in the compressed data and extract JSON columns

The data from Kaggle downloads into a .zip folder. If we were to extract everything in the .zip folder, we'd end up with over 30 GB - that's huge! So it's best to leave it zipped and use the `zipfile` library to handle reading individual files. We can use the `.open()` method to access the specific file we need when reading into pandas.

```python
z = ZipFile("ga-customer-revenue-prediction.zip")
```

The earlier version of the training data `train.csv` is a whopping 17 times smaller than the final version, `train_v2.csv`, so I'll use it for initial data exploration. Let's look at the columns in both versions to make sure the same columns are available in both.

```python
train_tmp = pd.read_csv(z.open('train_v2.csv'),nrows=100)
train_tmp.columns
```
<pre class="out">
Index(['channelGrouping', 'customDimensions', 'date', 'device',
       'fullVisitorId', 'geoNetwork', 'hits', 'socialEngagementType', 'totals',
       'trafficSource', 'visitId', 'visitNumber', 'visitStartTime'],
      dtype='object')
</pre>

```python
train_v1_tmp = pd.read_csv(z.open('train.csv'),nrows=100)
train_v1_tmp.columns
```
<pre class="out">
Index(['channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
       'sessionId', 'socialEngagementType', 'totals', 'trafficSource',
       'visitId', 'visitNumber', 'visitStartTime'],
      dtype='object')
</pre>

It looks like there are a few differences in columns between the earlier version and the final version of the training data. The earlier version does not include the columns `customDimensions` and `hits`, and the newer version does not include `sessionId`.

```python
train_tmp.sample(5) #examine v2 training data
```
<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>channelGrouping</th>
  <th>customDimensions</th>
  <th>date</th>
  <th>device</th>
  <th>fullVisitorId</th>
  <th>geoNetwork</th>
  <th>hits</th>
  <th>socialEngagementType</th>
  <th>totals</th>
  <th>trafficSource</th>
  <th>visitId</th>
  <th>visitNumber</th>
  <th>visitStartTime</th>
</tr>
</thead>
<tbody>
<tr>
  <th>66</th>
  <td>Organic Search</td>
  <td>[]</td>
  <td>20171016</td>
  <td>{"browser": "Chrome", "browserVersion": "not a...</td>
  <td>1858204879892046296</td>
  <td>{"continent": "Europe", "subContinent": "North...</td>
  <td>[{'hitNumber': '1', 'time': '0', 'hour': '4', ...</td>
  <td>Not Socially Engaged</td>
  <td>{"visits": "1", "hits": "3", "pageviews": "3",...</td>
  <td>{"campaign": "(not set)", "source": "google", ...</td>
  <td>1508152383</td>
  <td>1</td>
  <td>1508152383</td>
</tr>
<tr>
  <th>40</th>
  <td>Referral</td>
  <td>[{'index': '4', 'value': 'APAC'}]</td>
  <td>20171016</td>
  <td>{"browser": "Chrome", "browserVersion": "not a...</td>
  <td>8603113812592998660</td>
  <td>{"continent": "Asia", "subContinent": "Souther...</td>
  <td>[{'hitNumber': '1', 'time': '0', 'hour': '19',...</td>
  <td>Not Socially Engaged</td>
  <td>{"visits": "1", "hits": "3", "pageviews": "3",...</td>
  <td>{"referralPath": "/gopher", "campaign": "(not ...</td>
  <td>1508206196</td>
  <td>1</td>
  <td>1508206196</td>
</tr>
<tr>
  <th>61</th>
  <td>Referral</td>
  <td>[{'index': '4', 'value': 'EMEA'}]</td>
  <td>20171016</td>
  <td>{"browser": "Firefox", "browserVersion": "not ...</td>
  <td>2812958845084778564</td>
  <td>{"continent": "Europe", "subContinent": "South...</td>
  <td>[{'hitNumber': '1', 'time': '0', 'hour': '10',...</td>
  <td>Not Socially Engaged</td>
  <td>{"visits": "1", "hits": "3", "pageviews": "3",...</td>
  <td>{"referralPath": "/analytics/web/", "campaign"...</td>
  <td>1508174277</td>
  <td>1</td>
  <td>1508174277</td>
</tr>
<tr>
  <th>78</th>
  <td>Organic Search</td>
  <td>[{'index': '4', 'value': 'South America'}]</td>
  <td>20171016</td>
  <td>{"browser": "Chrome", "browserVersion": "not a...</td>
  <td>9325268255005246823</td>
  <td>{"continent": "Americas", "subContinent": "Sou...</td>
  <td>[{'hitNumber': '1', 'time': '0', 'hour': '3', ...</td>
  <td>Not Socially Engaged</td>
  <td>{"visits": "1", "hits": "3", "pageviews": "3",...</td>
  <td>{"campaign": "(not set)", "source": "google", ...</td>
  <td>1508151230</td>
  <td>1</td>
  <td>1508151230</td>
</tr>
<tr>
  <th>33</th>
  <td>Organic Search</td>
  <td>[{'index': '4', 'value': 'North America'}]</td>
  <td>20171016</td>
  <td>{"browser": "Safari", "browserVersion": "not a...</td>
  <td>953429090981710815</td>
  <td>{"continent": "Americas", "subContinent": "Nor...</td>
  <td>[{'hitNumber': '1', 'time': '0', 'hour': '15',...</td>
  <td>Not Socially Engaged</td>
  <td>{"visits": "1", "hits": "3", "pageviews": "3",...</td>
  <td>{"campaign": "(not set)", "source": "google", ...</td>
  <td>1508193746</td>
  <td>1</td>
  <td>1508193746</td>
</tr>
</tbody>
</table>
</div>


Here we can see that many of the columns actually contain JSON data fields within them. We need to flatten all the fields and extract the JSON information in order to get the data into a usable format. If we examine the fields closely, we can see that the `customDimensions` and `hits` columns have single-quoted values for the JSON field names, rather than the expected double quotes. For now let's leave out these columns, and we'll also leave out the `sessionId` column that only appears in the earlier version of the training data set.

```python
use_cols = ['channelGrouping',
        'date',
        'device',
        'fullVisitorId',
        'geoNetwork',
        'socialEngagementType',
        'totals',
        'trafficSource',
        'visitId',
        'visitNumber',
        'visitStartTime']
```

```python
def load_df(csv_path='', nrows=None):
json_cols = ['device','totals','geoNetwork','trafficSource']

df = pd.read_csv(csv_path,
                converters={column: json.loads for column in json_cols},
                dtype={'fullVisitorId':'str'},
                usecols = use_cols,
                nrows=nrows)

for column in json_cols:
    col_as_df = json_normalize(df[column])
    col_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in col_as_df.columns]
    df = df.drop(column, axis=1).merge(col_as_df, right_index=True, left_index=True)
return df
```
Now let's load in the smaller training data set and extract the JSON fields.

```python
%%time
train = load_df(csv_path=z.open('train.csv'))
```
<pre class="out">
Wall time: 2min
</pre>

```python
train.memory_usage(deep=True).sum() * 1e-6 #megabytes
```
<pre class="out">
2969.0272339999997
</pre>

```python
train.shape
```
<pre class="out">
(903653, 54)
</pre>

There are just over 900,000 rows, and right now it's using around 3 GB of memory. That's pretty large but still workable, but when we load in the final training data (a 24 GB file) that won't be pretty. Now that we've extracted all the JSON fields, let's look at the data.

### NAs and useless columns

```python
train.sample(5)
```

<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>channelGrouping</th>
  <th>date</th>
  <th>fullVisitorId</th>
  <th>socialEngagementType</th>
  <th>visitId</th>
  <th>visitNumber</th>
  <th>visitStartTime</th>
  <th>device.browser</th>
  <th>device.browserSize</th>
  <th>device.browserVersion</th>
  <th>...</th>
  <th>trafficSource.adwordsClickInfo.isVideoAd</th>
  <th>trafficSource.adwordsClickInfo.page</th>
  <th>trafficSource.adwordsClickInfo.slot</th>
  <th>trafficSource.campaign</th>
  <th>trafficSource.campaignCode</th>
  <th>trafficSource.isTrueDirect</th>
  <th>trafficSource.keyword</th>
  <th>trafficSource.medium</th>
  <th>trafficSource.referralPath</th>
  <th>trafficSource.source</th>
</tr>
</thead>
<tbody>
<tr>
  <th>646242</th>
  <td>Organic Search</td>
  <td>20161006</td>
  <td>2389916373011432409</td>
  <td>Not Socially Engaged</td>
  <td>1475787144</td>
  <td>1</td>
  <td>1475787144</td>
  <td>Chrome</td>
  <td>not available in demo dataset</td>
  <td>not available in demo dataset</td>
  <td>...</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not set)</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not provided)</td>
  <td>organic</td>
  <td>NaN</td>
  <td>google</td>
</tr>
<tr>
  <th>118983</th>
  <td>Organic Search</td>
  <td>20170421</td>
  <td>8165609625075911090</td>
  <td>Not Socially Engaged</td>
  <td>1492828420</td>
  <td>1</td>
  <td>1492828420</td>
  <td>Chrome</td>
  <td>not available in demo dataset</td>
  <td>not available in demo dataset</td>
  <td>...</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not set)</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not provided)</td>
  <td>organic</td>
  <td>NaN</td>
  <td>google</td>
</tr>
<tr>
  <th>170453</th>
  <td>Referral</td>
  <td>20170627</td>
  <td>8237815279985371728</td>
  <td>Not Socially Engaged</td>
  <td>1498595141</td>
  <td>1</td>
  <td>1498595141</td>
  <td>Chrome</td>
  <td>not available in demo dataset</td>
  <td>not available in demo dataset</td>
  <td>...</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not set)</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>referral</td>
  <td>/</td>
  <td>mall.googleplex.com</td>
</tr>
<tr>
  <th>37641</th>
  <td>Organic Search</td>
  <td>20170611</td>
  <td>7256651340144904184</td>
  <td>Not Socially Engaged</td>
  <td>1497247503</td>
  <td>4</td>
  <td>1497247503</td>
  <td>Chrome</td>
  <td>not available in demo dataset</td>
  <td>not available in demo dataset</td>
  <td>...</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not set)</td>
  <td>NaN</td>
  <td>True</td>
  <td>(not provided)</td>
  <td>organic</td>
  <td>NaN</td>
  <td>google</td>
</tr>
<tr>
  <th>430408</th>
  <td>Organic Search</td>
  <td>20170214</td>
  <td>3505339230161918254</td>
  <td>Not Socially Engaged</td>
  <td>1487078543</td>
  <td>1</td>
  <td>1487078543</td>
  <td>Chrome</td>
  <td>not available in demo dataset</td>
  <td>not available in demo dataset</td>
  <td>...</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not set)</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>(not provided)</td>
  <td>organic</td>
  <td>NaN</td>
  <td>google</td>
</tr>
</tbody>
</table>
<p>5 rows × 54 columns</p>
</div>


We can see that there are many fields that have missing information, for example the `device.browserSize` field which has "not available in demo dataset" in every row. Let's convert all these missing value indicator strings into actual NAs and see if there are columns that only contain missing/NA values.

```python
na_list = ['(not set)',
'not available in demo dataset',
'(not provided)',
'(none)',
'unknown.unknown']
```

```python
#convert list of values above to NA
#check if columns only contain NA values and drop from data frame
na_cols(train, na_vals=na_list, inplace=True, drop=True)
```
<pre class="out">
The following columns contain only NA values and have been dropped:
 ['device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'trafficSource.adwordsClickInfo.criteriaParameters']
</pre>

Whew, looks like we got rid of a lot of dead weight! Now let's see if there are any columns where every row has the same value.


```python
one_val = one_val_cols(train)
```
<pre class="out">
The following columns contain only 1 unique value:

socialEngagementType  :  Not Socially Engaged
totals.visits  :  1
</pre>

Looks like we can drop some of these columns, since they don't contain any useful information (for example, `socialEngagmentType`). Since we've saved the list of column names into the variable `one_val`, we can easily drop the columns from the data frame.


```python
train.drop(columns=one_val, inplace=True)
```

Now that we've removed a number of columns, it looks like we're only using about half the memory of the original data frame. We can reduce that even more by choosing more appropriate data types, since the majority of the columns are currently `object` type (pandas' version of strings), which are very inefficient.


```python
train.memory_usage(deep=True).sum() * 1e-6 #megabytes
```
<pre class="out">
1368.347474
</pre>

```python
train.dtypes.value_counts()
```
<pre class="out">
object    30
int64      4
bool       1
</pre>

### Save memory by converting data types

By converting one of the object columns to pandas' native categorical type, we can see how much more efficient it is. For this column, the size was reduced from 57 MB to less than 1 MB.

```python
print('size before:', train['device.deviceCategory'].memory_usage(deep=True) * 1e-6)
train['device.deviceCategory'] = train['device.deviceCategory'].astype('category')
print('size after :', train['device.deviceCategory'].memory_usage(deep=True) * 1e-6)
```
<pre class="out">
size before: 57.59
size after : 0.90
</pre>

Now let's convert the rest of the columns to the proper data types, including the `date` and `visitStartTime` columns. For the numeric columns, we can use `to_numeric()` with the `downcast` argument to save space. This will find the smallest data type needed to store the column's values and change it to that type. For example, a column that is `int64` but only contains numbers that go up to the hundreds or thousands will be converted to `int16`.

```python
train['date'] = pd.to_datetime(train['date'],format='%Y%m%d').dt.date
train['visitStartTime'] = pd.to_datetime(train['visitStartTime'],format='%H%M%S%f').dt.time
```

```python
cat_cols = ['channelGrouping', 'device.browser', 'device.deviceCategory', 'device.operatingSystem',
        'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.continent',
       'geoNetwork.region', 'geoNetwork.subContinent', 'trafficSource.medium']
for col in cat_cols:
train[col] = train[col].astype('category')
```

```python
num_cols = ['visitId','visitNumber','totals.transactionRevenue','totals.hits','totals.pageviews']

for col in num_cols:
train[col] = pd.to_numeric(train[col], downcast='integer')
```

### Examine the remaining columns

Now that we've cleaned up the column types, let's examine the data a little more closely. We'll start by looking at the fields extracted from the original "device" JSON column.

```python
#look at device JSON fields
train.filter(like='device', axis=1).sample(5)
```

<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>device.browser</th>
  <th>device.deviceCategory</th>
  <th>device.isMobile</th>
  <th>device.operatingSystem</th>
</tr>
</thead>
<tbody>
<tr>
  <th>6256</th>
  <td>Chrome</td>
  <td>desktop</td>
  <td>False</td>
  <td>Windows</td>
</tr>
<tr>
  <th>7</th>
  <td>Firefox</td>
  <td>desktop</td>
  <td>False</td>
  <td>Macintosh</td>
</tr>
<tr>
  <th>49032</th>
  <td>Chrome</td>
  <td>desktop</td>
  <td>False</td>
  <td>Windows</td>
</tr>
<tr>
  <th>179013</th>
  <td>UC Browser</td>
  <td>desktop</td>
  <td>False</td>
  <td>Linux</td>
</tr>
<tr>
  <th>801229</th>
  <td>Chrome</td>
  <td>mobile</td>
  <td>True</td>
  <td>Android</td>
</tr>
</tbody>
</table>
</div>



We only have four device-related columns after dropping the NA columns. We'll keep all of them for now since they may be useful, and we aren't doing a full EDA on the data yet. Next let's look at the "geoNetwork" JSON fields.


```python
train.filter(like='geo', axis=1).sample(5)
```


<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>geoNetwork.city</th>
  <th>geoNetwork.continent</th>
  <th>geoNetwork.country</th>
  <th>geoNetwork.metro</th>
  <th>geoNetwork.networkDomain</th>
  <th>geoNetwork.region</th>
  <th>geoNetwork.subContinent</th>
</tr>
</thead>
<tbody>
<tr>
  <th>280357</th>
  <td>Izmir</td>
  <td>Asia</td>
  <td>Turkey</td>
  <td>NaN</td>
  <td>ttnet.com.tr</td>
  <td>Izmir</td>
  <td>Western Asia</td>
</tr>
<tr>
  <th>385289</th>
  <td>NaN</td>
  <td>Oceania</td>
  <td>Australia</td>
  <td>NaN</td>
  <td>dodo.net.au</td>
  <td>NaN</td>
  <td>Australasia</td>
</tr>
<tr>
  <th>733039</th>
  <td>Madrid</td>
  <td>Europe</td>
  <td>Spain</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>Community of Madrid</td>
  <td>Southern Europe</td>
</tr>
<tr>
  <th>318694</th>
  <td>NaN</td>
  <td>Asia</td>
  <td>Indonesia</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>Southeast Asia</td>
</tr>
<tr>
  <th>639501</th>
  <td>NaN</td>
  <td>Europe</td>
  <td>United Kingdom</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>NaN</td>
  <td>Northern Europe</td>
</tr>
</tbody>
</table>
</div>

From a quick look here, we have many fields that may have inconsistently available information or not have any information at all. The fields that look like they may be useful here are continent, country, region, and sub-continent. Region might be too granular (in addition to containing NAs), whereas continent may not be granular enough. Country and sub-continent are probably the most useful fields to us here, but for now we can keep most of the fields in until we can do a proper EDA.

```python
geo_drop_cols = ['geoNetwork.city', 'geoNetwork.networkDomain']
```
Next up are the "traffic" columns. Most of these columns looked like they were full of NAs, so let's run a `describe()` instead of taking a sample.

```python
train.filter(like='traffic', axis=1).describe()
```

<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>trafficSource.adContent</th>
  <th>trafficSource.adwordsClickInfo.adNetworkType</th>
  <th>trafficSource.adwordsClickInfo.gclId</th>
  <th>trafficSource.adwordsClickInfo.isVideoAd</th>
  <th>trafficSource.adwordsClickInfo.page</th>
  <th>trafficSource.adwordsClickInfo.slot</th>
  <th>trafficSource.campaign</th>
  <th>trafficSource.campaignCode</th>
  <th>trafficSource.isTrueDirect</th>
  <th>trafficSource.keyword</th>
  <th>trafficSource.medium</th>
  <th>trafficSource.referralPath</th>
  <th>trafficSource.source</th>
</tr>
</thead>
<tbody>
<tr>
  <th>count</th>
  <td>10946</td>
  <td>21460</td>
  <td>21561</td>
  <td>21460</td>
  <td>21460</td>
  <td>21460</td>
  <td>38306</td>
  <td>1</td>
  <td>274005</td>
  <td>34361</td>
  <td>760507</td>
  <td>330941</td>
  <td>903584</td>
</tr>
<tr>
  <th>unique</th>
  <td>44</td>
  <td>2</td>
  <td>17774</td>
  <td>1</td>
  <td>8</td>
  <td>2</td>
  <td>9</td>
  <td>1</td>
  <td>1</td>
  <td>3658</td>
  <td>5</td>
  <td>1475</td>
  <td>379</td>
</tr>
<tr>
  <th>top</th>
  <td>Google Merchandise Collection</td>
  <td>Google Search</td>
  <td>Cj0KEQjwmIrJBRCRmJ_x7KDo-9oBEiQAuUPKMufMpuG3Zd...</td>
  <td>False</td>
  <td>1</td>
  <td>Top</td>
  <td>Data Share Promo</td>
  <td>11251kjhkvahf</td>
  <td>True</td>
  <td>6qEhsCssdK0z36ri</td>
  <td>organic</td>
  <td>/</td>
  <td>google</td>
</tr>
<tr>
  <th>freq</th>
  <td>5122</td>
  <td>21453</td>
  <td>70</td>
  <td>21460</td>
  <td>21362</td>
  <td>20956</td>
  <td>16403</td>
  <td>1</td>
  <td>274005</td>
  <td>11503</td>
  <td>381561</td>
  <td>75523</td>
  <td>400788</td>
</tr>
</tbody>
</table>
</div>

It looks like some of the `trafficSource` fields might be more useful than others. For example, the campaign and campaign codes will likely change over time, meaning they aren't likely to be good features. Others, like `adContent`, have only a tiny fraction of non-null values. Since we have so many other features to choose from, we can probably drop some of these.

```python
traffic_drop_cols = ['trafficSource.adwordsClickInfo.gclId','trafficSource.campaign',
                 'trafficSource.campaignCode', 'trafficSource.adwordsClickInfo.adNetworkType',
                'trafficSource.referralPath', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.page',
                'trafficSource.adwordsClickInfo.slot']
train.drop(columns = traffic_drop_cols, inplace=True)
```

Finally, let's look at the "totals" columns, which includes our target variable, revenue.

```python
train.filter(like='total', axis=1).sample(5)
```

<div class="overflow">
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>totals.bounces</th>
  <th>totals.hits</th>
  <th>totals.newVisits</th>
  <th>totals.pageviews</th>
  <th>totals.transactionRevenue</th>
</tr>
</thead>
<tbody>
<tr>
  <th>61806</th>
  <td>1</td>
  <td>1</td>
  <td>1</td>
  <td>1.0</td>
  <td>NaN</td>
</tr>
<tr>
  <th>74190</th>
  <td>1</td>
  <td>1</td>
  <td>1</td>
  <td>1.0</td>
  <td>NaN</td>
</tr>
<tr>
  <th>510062</th>
  <td>1</td>
  <td>1</td>
  <td>1</td>
  <td>1.0</td>
  <td>NaN</td>
</tr>
<tr>
  <th>219742</th>
  <td>1</td>
  <td>1</td>
  <td>1</td>
  <td>1.0</td>
  <td>NaN</td>
</tr>
<tr>
  <th>690457</th>
  <td>1</td>
  <td>1</td>
  <td>NaN</td>
  <td>1.0</td>
  <td>NaN</td>
</tr>
</tbody>
</table>
</div>

Finally, let's examine the revenue column, since we'll be using that to calculate the target variable.

```python
print('There are', train.shape[0],'rows and',
  train['totals.transactionRevenue'].isnull().sum(), 'nulls.')
```
<pre class="out">
There are 903653 rows and 892138 nulls.
</pre>

```python
train['totals.transactionRevenue'].describe()
```
<pre class="out">
count    1.151500e+04
mean     1.337448e+08
std      4.482852e+08
min      1.000000e+04
25%      2.493000e+07
50%      4.945000e+07
75%      1.076550e+08
max      2.312950e+10
</pre>

These number seem way too large to be dollars, so let's convert them to dollars and fill in the NAs with 0 (since NA means there was no purchase).

```python
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0.0)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].astype('float')
train['totals.transactionRevenue'] = train['totals.transactionRevenue'] / 1000000
```

Now we've cut down the amount of columns we'll keep in for our EDA, and we've converted the column types to save memory. The next step is to load and transform the full training and test data sets we'll be using, then save them as feather files for quick loading in the future.

Here are the steps we've taken so far:

- Read in the zip file and extract JSON fields into separate columns
- Convert string values like "not set" to NA and drop columns that contain only NAs
- Drop columns that contain only one value and are therefore not informative
- Examine the remaining columns and drop those that do not seem useful
- Convert data types to more appropriate and efficient types

We don't need to re-examine all the columns in the training and test sets, since we already know which columns to drop. So we can skip that step, but we'll do all the other steps.

### Prepare the test data

We'll read in the data and extract the JSON fields using the function we defined above.

```python
use_cols = ['channelGrouping',
        'date',
        'device',
        'fullVisitorId',
        'geoNetwork',
        'totals',
        'trafficSource',
        'visitId',
        'visitNumber',
        'visitStartTime']
```


```python
%%time
test = load_df(csv_path=z.open('test_v2.csv'))
```
<pre class="out">
Wall time: 1min 36s
</pre>


```python
test.memory_usage(deep=True).sum() * 1e-6 #megabytes
```

<pre class="out">
1371.997621
</pre>

Now we'll drop the unneeded columns and convert the NAs.


```python
drop_cols = ['totals.visits', 'trafficSource.adwordsClickInfo.gclId','trafficSource.campaign',
         #'trafficSource.campaignCode',
         'trafficSource.adwordsClickInfo.adNetworkType',
         'trafficSource.referralPath', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.page',
         'trafficSource.adwordsClickInfo.slot', 'geoNetwork.city', 'geoNetwork.networkDomain']
test.drop(columns=drop_cols, inplace=True)
```


```python
na_cols(test, na_vals=na_list, inplace=True, drop=True)
```
<pre class="out">
The following columns contain only NA values and have been dropped:
 ['device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'trafficSource.adwordsClickInfo.criteriaParameters']
</pre>

Next, we'll fix the data types.

```python
test['date'] = pd.to_datetime(test['date'],format='%Y%m%d').dt.date
test['visitStartTime'] = pd.to_datetime(test['visitStartTime'],format='%H%M%S%f').dt.time
```


```python
for col in cat_cols:
test[col] = test[col].astype('category')
```


```python
for col in num_cols:
test[col] = pd.to_numeric(test[col], downcast='integer')
```


```python
test['totals.transactionRevenue'] = test['totals.transactionRevenue'].fillna(0.0)
test['totals.transactionRevenue'] = test['totals.transactionRevenue'].astype('float')
test['totals.transactionRevenue'] = test['totals.transactionRevenue'] / 1000000
```


```python
test.memory_usage(deep=True).sum() * 1e-6 #megabytes
```

<pre class="out">
255.960481
</pre>

Finally, we'll save as a `feather` file so we can load it in quickly when we do our EDA.


```python
test.to_feather('test.feather')
```

```python
print("test.feather:", os.stat('test.feather').st_size * 1e-6)
```
<pre class="out">
test.feather: 23.307498
</pre>

### Prepare the training data

Here's the real test - this file is almost 24 GB so let's see if we can handle reading in the data frame and extracting the JSON fields.

```python
%%time
train = load_df(csv_path=z.open('train_v2.csv'))
```
<pre class="out">
Wall time: 6min 23s
</pre>

```python
train.memory_usage(deep=True).sum() * 1e-6 #megabytes
```
<pre class="out">
5751.213158
</pre>

Not bad! Now let's get rid of some of the dead weight columns and fix the NAs.

```python
drop_cols = ['totals.visits', 'trafficSource.adwordsClickInfo.gclId','trafficSource.campaign',
         'trafficSource.campaignCode', 'trafficSource.adwordsClickInfo.adNetworkType',
         'trafficSource.referralPath', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.page',
         'trafficSource.adwordsClickInfo.slot', 'geoNetwork.city', 'geoNetwork.networkDomain']
train.drop(columns=drop_cols, inplace=True)
```


```python
na_cols(train, na_vals=na_list, inplace=True, drop=True)
```
<pre class="out">
The following columns contain only NA values and have been dropped:
 ['device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'trafficSource.adwordsClickInfo.criteriaParameters']
</pre>

We'll convert the data types and slim down our data frame size even more.


```python
train['date'] = pd.to_datetime(train['date'],format='%Y%m%d').dt.date
train['visitStartTime'] = pd.to_datetime(train['visitStartTime'],format='%H%M%S%f').dt.time
```


```python
for col in cat_cols:
train[col] = train[col].astype('category')
```


```python
for col in num_cols:
train[col] = pd.to_numeric(train[col], downcast='integer')
```


```python
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0.0)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].astype('float')
train['totals.transactionRevenue'] = train['totals.transactionRevenue'] / 1000000
```


```python
train.memory_usage(deep=True).sum() * 1e-6 #megabytes
```


<pre class="out">
1070.337347
</pre>

Finally, as above, we'll save our newly svelte training data to a `feather` file for easy loading next time.


```python
train.to_feather('train.feather')
```


```python
print("train.feather:", os.stat('train.feather').st_size * 1e-6)
```
<pre class="out">
train.feather: 94.12897799999999
</pre>

### Conclusion

Whew, we did it! We started with some very large files that would have been difficult to work with and got them down to a much more manageable size. Keep in mind, we were able to manage this even on my not-particularly-powerful personal desktop computer, so these strategies are quite useful. Now we can just load in the (very small) feather files and perform our EDA, feature engineering, and modeling in a new notebook without having to deal with these enormous data files.

To recap, some helpful strategies to use when dealing with large files or large data sets include:

- Using the `zipfile` library to read in compressed files without having to unzip them. 
- Converting the columns to more efficient data types. Strings (pandas' `object` type) are usually the biggest memory hogs, so that's a good place to start.
- Saving the data as a feather file once preprocessing is finished.
