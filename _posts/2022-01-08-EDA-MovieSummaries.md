---
published: false
---
## MultiLabel Classification

# MultiLabel Classification

Consider a situation in which you go to the theater, you want to watch a movie which is both Adventure and Comedy. In this case if the **usher** (like the cashier in the theater) asks you to only choose one of those genres then the movie probably wouldn’t be good (atleast for you). 

The ability to choose something which isn’t **mutually exclusive** is multilabel classification.

**Mutually Exclusive :** Heads or Tails. You cannot have both Heads and Tails at the same toss of coin.

Movie Genres are not mutually exclusive, you can have a movie which is Adventurous, Comedy and Historic, all at the same time. This is the reason we will be using MovieSummaries Dataset to learn about MultiLabel Classification. 

This is a follow along demonstration of MultiLabel Classification. This tutorial assumes basic knowledge of Python, Pandas, Visualisation libraries.

This is part 1 of MultiLabel Classification which performs EDA on the dataset.

# MovieSummaries Dataset:

The dataset can be downloaded from the following [link](http://www.cs.cmu.edu/~ark/personas/).

This dataset contains 42,306 movie plot summaries extracted from Wikipedia along with metadata extracted from FreeBase. We will be using only two files : `movie.metadata.tsv` and `plot_summaries.txt` . 

You can also use my `!gdown --id` code if you don’t want to download the whole dataset in your local computer.

# GamePlan:

## 1. Downloading the Data Set

## 2. Reading the Data Set

## 3. Exploratory Data Analysis (EDA)

## Let’s Get Started

# 1. Downloading the Data Set:

We download the dataset from the Drive Link using `!gdown` . We provide the id and colab automatically downloads the dataset for us and it is visible in the files section.

```python
!gdown --id 1fD9_t_EFOYTe4PXh4V9WTsKqHBL1z6VR
!gdown --id 1LmO50zgMg_zg-cNWLGIPO5A4WROg-6BP
```

We need to import necessary libraries for performing EDA. 

We will be importing the following modules:

- Pandas - For data manupalation and analysis
- Numpy - For array manipulation
- Matplotlib.pyplot - For plotting graphs and visualizing data
- Seaborn - For high level visualization
- csv - For reading and writing CSV files
- tqdm - Instantly make your loops show a smart progress meter
- json - to store and transfer json format data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from tqdm import tqdm
import seaborn as sns
```

# 2. Reading the Data Set:

```python
genre = pd.read_table('/content/movie.metadata.tsv',header = None)
genre.head()
```

If you see the code, you can see that the data file is of format **tsv** . It stands for Tab Seperated Value and all the records are seperated by tab. 



We can see that we get few columns which don’t provide us with any meaning, so we are going to **drop** them. 

```python
genre = genre.drop([1,3,4,5,6,7],axis = 1)
genre.head()
```


Now that we have removed the useless columns, let us name the other columns. 

```python
column_name = ['Movie_Id','Movie_Name','Movie_Genres']
genre.columns = column_name
genre.head()
```



We have downloaded the tsv file, which is the easy part. Now we need to download the txt file and convert it into a **Pandas DataFrame**. The following code achieves that:

```python
file_path = '/content/plot_summaries.txt'
plot_f = []
with open(file_path, 'r') as f:
    r = csv.reader(f, dialect='excel-tab') 
    for row in r:
        plot_f.append(row)

movie_plot = []
movie_id = []

for i in plot_f:
  movie_plot.append(i[1])
  movie_id.append(i[0])

plot = pd.DataFrame({'Movie_Id': movie_id, 'Movie_Plot': movie_plot})
plot.head()
```

In this code snippet, we can see that first we open the text file, get the movie_id and movie_plot and add it to a python list named `plot_f` . We split the list into seperate movie_id and movie_plot, which is then used to make a Pandas DataFrame.

The `Movie_Id` variable may be of type float, we are going to convert them into String type to perform `merge` function.

```python
genre['Movie_Id'] = genre['Movie_Id'].astype(str)
plot['Movie_Id'] = plot['Movie_Id'].astype(str)
genre['Movie_Id']
plot['Movie_Id']
```

We can see that in both the tables, `Movie_Id` is of type object, which is what we need.



Now, we can merge both of these tables.

```python
final_df = pd.merge(genre,plot,on = "Movie_Id")
final_df.head()
```

# 3. Exploratory Data Analysis

Now we get to the interesting part of the whole tutorial. We are going to get the genre types which, as we can see, are inside a dictionary of sorts. 

We can take the values outside using `json.loads()` .

```python
json.loads(final_df['Movie_Genres'][0]).values()
```

Now, we will create a variable called `movie_genres_updated` which will go through the `final_df['Movie_Genres']` and converts all the dictionary values (json) into lists.

```python
movie_genres_updated = []
for i in final_df['Movie_Genres']:
  movie_genres_updated.append(list(json.loads(i).values()))
final_df['Movie_Genres'] = movie_genres_updated
final_df.head()
```

Now all our genres are in a list, but we cannot perform **EDA** without knowing the number of times a Genre repeats. So we are going to remove the list and create a dictionary with the key values being the **frequency** of genres. 

```python
total_genres = sum(movie_genres_updated,[])
unique_genres = set(total_genres)
len(unique_genres)
```

```python
genres_dict = {}
for i in tqdm(unique_genres):
  count = 0
  for j in total_genres:
    if j == i:
      count += 1
  genres_dict[i] = count
```

We got the frequency distribution of the genres, now we can perform **EDA.** 

Now, we are going to create 4 functions:

1. **n_largest_dict_maker** : Creates the n largest frequency of genres
2. **n_smallest_dict_maker** : Creates the n smallest frequence of genres
3. **pieChartMake**r : For creating a Pie Chart
4. **barPlotMaker** :  For creating a bar plot

```python
def n_largest_dict_maker(dict_name,n_value):
  '''
  This function takes two inputs : a dataframe and n_value. It return top n_value of 
  labels and their frequencies
  '''
  n_largest_dict = sorted(dict_name.values(),reverse = True)
  l = n_largest_dict[n_value]
  labels = []
  sizes = []

  for x, y in genres_dict.items():
    if y > l:
      labels.append(x)
      sizes.append(y)
  return labels, sizes

def n_smallest_dict_maker(dict_name,n_value):
  '''
  This function takes two inputs : a dataframe and n_value. It return bottom n_value of 
  labels and their frequencies
  '''
  n_largest_dict = sorted(dict_name.values(),reverse = True)
  l = n_largest_dict[-1 * n_value]
  labels = []
  sizes = []

  for x, y in genres_dict.items():
    if y < l:
      labels.append(x)
      sizes.append(y)
  return labels, sizes

def pieChartMaker(labels,sizes):
  '''
  This function creates a pie chart from labels(genres) and their sizes(frequencies)
  '''
  colors = sns.color_palette('pastel')[0:5]
  plt.pie(sizes, labels=labels, colors = colors)
  plt.axis('equal')
  plt.show()

def barPlotMaker(labels,sizes):
  '''
  This function creates a bar plot from labels(genres) and their sizes(frequencies)
  '''
  n_df = pd.DataFrame({"Labels" : labels,"Sizes" : sizes})
  plt.figure(figsize=(12,15))
  ax = sns.barplot(data=n_df, x= "Sizes", y = "Labels")
  ax.set(ylabel = 'Sizes')
  plt.show()
```

That’s it, we can visualize the distribution of genres. We will look at the following things:

1. Bar Plot of Top 30 Genres
2. Bar Plot of Bottom 30 Genres
3. Pie Chart of Bottom 30 Genres

## 1. Bar Plot of Top 30 Genres

```python
labels,sizes = n_largest_dict_maker(genres_dict,30)
barPlotMaker(labels,sizes)
```



## 2. Bar Plot of Bottom 30 Genres

```python
labels,sizes = n_smallest_dict_maker(genres_dict,30)
barPlotMaker(labels,sizes)
```



## 3. Pie Chart of Top 20 Genres

```python
labels,sizes = n_largest_dict_maker(genres_dict,20)
pieChartMaker(labels,sizes)
```



That completes our EDA for MovieSummaries Dataset. This was surely interesting to code, share your comments in the comments section. Reach out if you have any queries. 

The github link to this code along can be found [here](https://github.com/Mukilan-Krishnakumar/MultiLabelClassification_MovieSummaries/blob/main/EDA_of_MovieSummaries.ipynb).

Most of the code snippets were inspired from this [link](https://colab.research.google.com/github/prateekjoshi565/movie_genre_prediction/blob/master/Movie_Genre_Prediction.ipynb#scrollTo=ljJeSX0hF2v7).
