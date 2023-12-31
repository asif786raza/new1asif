# Problem Statement :

- Create a Recommender System to show personalized movie recommendations based on ratings given by a user and other users similar to them in order to improve user experience.
---
# Data Dictionary:

### RATINGS FILE DESCRIPTION

- All ratings are contained in the file "ratings.dat" and are in the following format:

      - UserID::MovieID::Rating::Timestamp

      - UserIDs range between 1 and 6040

      - MovieIDs range between 1 and 3952

      - Ratings are made on a 5-star scale (whole-star ratings only)

      - Timestamp is represented in seconds

      - Each user has at least 20 ratings


### USERS FILE DESCRIPTION

- User information is in the file "users.dat" and is in the following format:

    - UserID::Gender::Age::Occupation::Zip-code

- All demographic information is provided voluntarily by the users and is not checked for accuracy.
Only users who have provided some demographic information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female

      Age is chosen from the following ranges:

      1: "Under 18"
      18: "18-24"
      25: "25-34"
      35: "35-44"
      45: "45-49"
      50: "50-55"
      56: "56+"

- Occupation is chosen from the following choices:

      0: "other" or not specified
      1: "academic/educator"
      2: "artist"
      3: "clerical/admin"
      4: "college/grad student"
      5: "customer service"
      6: "doctor/health care"
      7: "executive/managerial"
      8: "farmer"
      9: "homemaker"
      10: "K-12 student"
      11: "lawyer"
      12: "programmer"
      13: "retired"
      14: "sales/marketing"
      15: "scientist"
      16: "self-employed"
      17: "technician/engineer"
      18: "tradesman/craftsman"
      19: "unemployed"
      20: "writer"

### MOVIES FILE DESCRIPTION

- Movie information is in the file "movies.dat" and is in the following format:

     - MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including year of release)

      - Genres are pipe-separated and are selected from the following genres:

      Action
      Adventure
      Animation
      Children's
      Comedy
      Crime
      Documentary
      Drama
      Fantasy
      Film-Noir
      Horror
      Musical
      Mystery
      Romance
      Sci-Fi
      Thriller
      War
      Western
# pip install gdown
# !pip install seaborn --upgrade
# !pip install matplotlib --upgrade


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignores FutureWarnings
warnings.simplefilter(action='ignore', category=UserWarning) 




# from google.colab import drive
# drive.mount('/content/drive')

movies = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-movies.dat",encoding="ISO-8859-1")
ratings =pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-ratings.dat",encoding="ISO-8859-1")
users = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-users.dat",encoding="ISO-8859-1")
delimiter ="::"

users = users["UserID::Gender::Age::Occupation::Zip-code"].str.split(delimiter,expand = True)
users.columns = ["UserID","Gender","Age","Occupation","Zipcode"]

users["Age"].replace({"1": "Under 18","18": "18-24","25": "25-34",
                          "35": "35-44","45": "45-49","50": "50-55","56": "56+"},inplace=True)

users["Occupation"] = users["Occupation"].astype(int).replace({0: "other",1: "academic/educator",2: "artist",
                                                               3: "clerical/admin",4: "college/grad student",
                                             5: "customer service",6: "doctor/health care",7: "executive/managerial",
                                             8: "farmer" ,9: "homemaker",10: "K-12 student",11: "lawyer",
                                             12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",
                                             16: "self-employed",17: "technician/engineer",
                                             18: "tradesman/craftsman",19: "unemployed",20: "writer"},
                                            )

delimiter ="::"

ratings = ratings["UserID::MovieID::Rating::Timestamp"].str.split(delimiter,expand = True)
ratings.columns = ["UserID","MovieID","Rating","Timestamp"]


movies.drop(["Unnamed: 1","Unnamed: 2"],axis = 1,inplace=True)



delimiter ="::"

movies = movies["Movie ID::Title::Genres"].str.split(delimiter,expand = True)
movies.columns = ["MovieID","Title","Genres"]


movies.shape,ratings.shape,users.shape

movies # need to take care of Genres .
ratings # need to convert timestamp to hrs.
users
# taking out the release year from the title column from movie table :

movies["Release_year"] = movies["Title"].str.extract('^(.+)\s\(([0-9]*)\)$',expand = True)[1]
movies["Title"] = movies["Title"].str.split("(").apply(lambda x:x[0])


# Converting timestamp to hours

from datetime import datetime
ratings["Watch_Hour"] =ratings["Timestamp"].apply(lambda x:datetime.fromtimestamp(int(x)).hour)
ratings.drop(["Timestamp"],axis = 1,inplace=True)

movies.shape,ratings.shape,users.shape

#### Merging all the tables into one data frame :
df = users.merge(movies.merge(ratings,on="MovieID",how="outer"),on="UserID",how="outer")
df.shape
df
df_ = df.copy()
df_.dropna(inplace=True)
df_.info()
df_['Release_year']=df_['Release_year'].astype('int32')
df_['Rating']=df_['Rating'].astype('int32')

bins = [1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 2000]
labels = ['20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

df_["Released_In"] =  pd.cut(df_['Release_year'], bins=bins, labels=labels)
import seaborn as sns
## Average user rating distribution :
# !pip install numpy pandas matplotlib seaborn
import matplotlib.pyplot as plt

# Group by 'UserID', calculate the mean, and then extract the 'Rating' column
avg_ratings = df_[['UserID', 'Rating']].groupby('UserID').mean()["Rating"]

# Plot the histogram using Matplotlib
plt.hist(avg_ratings, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Average Ratings by User')
plt.xlabel('Average Rating')
plt.ylabel('Number of Users')
plt.show()
# average ratings given by each user distribution


# Group by 'MovieID', calculate the mean, and then extract the 'Rating' column
avg_movie_ratings = df_[['MovieID', 'Rating']].groupby('MovieID').mean()["Rating"]

# Plot the histogram using Matplotlib
plt.figure(figsize=(10, 5))
plt.hist(avg_movie_ratings, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Average Ratings Received by Movies')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')

plt.tight_layout()
plt.show()


# average rating , that each movie has receieved by users .
df_["MovieID"].nunique()
movies_per_decade = df_[['MovieID','Released_In']].groupby('Released_In').nunique()
movies_per_decade["% of all Movies"] = (movies_per_decade["MovieID"]/(df_["MovieID"].nunique())) * 100
movies_per_decade
sns.barplot(x=movies_per_decade.index, y=movies_per_decade["% of all Movies"])



m = movies[["MovieID","Title","Genres"]]
m["Genres"] = m["Genres"].str.split("|")
m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })

m = m.pivot_table(values="Title", index="MovieID", columns="Genres", aggfunc= np.size,).fillna(0)


def apply(x):
  if x >= 1:
    return 1
  else:
    return 0

m["Adventure"] = m["Adventure"].apply(apply)
m = m.astype(int)
m
final_data = df.merge(m,on="MovieID",how="left").drop(["Genres"],axis = 1)
final_data

final_data.MovieID = final_data.MovieID.astype(int)
final_data.UserID = final_data.UserID.astype(float)
final_data.Release_year = final_data.Release_year.astype(float)
final_data.info()
final_data.describe()
final_data.describe(include="object")
final_data.nunique()
---
#### Unique values present in data
---
- 6040 unique UserID
- 7 different age groups
- 21 occupations
- 3439 different locations of users
-3883 unique movies

---
- There are movies available in database , which were never been watched by any user before .
- Thats is the reason we have lots of NaN values in our final dataset.
---
final_data.shape
plt.rcParams["figure.figsize"] = (20,8)

## Most of the movies present in our dataset were released in year:

final_data.groupby("Release_year")["Title"].nunique().plot(kind="bar")

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'm' is a DataFrame where each column represents a genre and each row represents a movie,
# and the entry is 1 if the movie belongs to that genre and 0 otherwise

# Number of Movies per Genres:
genres_counts = m.sum(axis=0)
sns.barplot(x=genres_counts.index, y=genres_counts.values)

plt.xticks(rotation=90)  # Rotate genre names for better visibility if needed
plt.ylabel('Number of Movies')
plt.title('Number of Movies per Genres')
plt.show()

m.sum(axis= 0)
final_data["Rating"].count()
## Number of movies Rated by each Gender type :
# Gender

asd = final_data.groupby("Gender")["Rating"].count() / final_data["Rating"].count() * 100
asd


plt.pie(asd, labels = ["Female", "Male"],autopct='%1.1f%%')

## Users of which age group have watched and rated the most number of movies?
plt.rcParams["figure.figsize"] = (10,6)
final_data.groupby("Age")["UserID"].nunique().plot(kind="bar")
- in DataSet : majority of the viewers are  in age group of 25-34
- out of all , 25-34 age group have rated and watched the maximum number of movies.
- for other age groups data are as below:
final_data.groupby("Age")["MovieID"].nunique()
plt.rcParams["figure.figsize"] = (10,8)
final_data.groupby("Age")["MovieID"].nunique().plot(kind="bar")

## Users belonging to which profession have watched and rated the most movies?



plt.rcParams["figure.figsize"] = (20,8)

plt.subplot(121)
final_data.groupby("Occupation")["UserID"].nunique().sort_values().plot(kind="bar")
plt.subplot(122)
final_data.groupby("Occupation")["MovieID"].nunique().sort_values().plot(kind="bar")

- Majority of the Users are College Graduates and Students , followed by Executives, educators and engineers.
y of the Users are College Graduates and Students , followed by Executives, educators and engineers.
- Maximum movies are watched and rated by user's occupations are College graduate students , writers , executives, educator and artists.
final_data.groupby("Occupation")["MovieID"].nunique().sort_values(ascending = False).head(6)
final_data.columns
## Movie Recommendation based on Genres as per Majority Users occupation :     
- below table shows the rank preference of each occupation users:
- higher the number more prefered .
## Movie Recommendation based on Genre as per Majority Users :
np.argsort((final_data.groupby("Occupation")['Action', 'Adventure', 'Animation', "Children's",
                                             'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy',
                                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Other',
                                             'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'].mean())   *100,axis = 1).loc[["writer","artist","academic/educator","executive/managerial","college/grad student"]]
- Writers , artists and educator most preferes to watch Animation, Fantasy and Science Fiction movies, followed by Romance , Action and rest of the genres.

- COllege Students most prefer to watch Children's , Science Fiction, Romance and Fantasy movies.

- Film-Noir is more prefered by the educators and Executive occupation users.


## what is the traffic on OTT, based on watch hour :

final_data.groupby("Watch_Hour")["UserID"].nunique().plot(kind="bar")


## Top 10 Movies have got the most number of ratings :
top10_movies = final_data.groupby("Title")["Rating"].count().reset_index().sort_values(by="Rating",ascending=False).head(10)
top10_movies
sns.barplot(y = top10_movies["Title"],
            x = top10_movies["Rating"])

## 5 Top rated Recommended Movies per each genre :
Genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime','Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Other','Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

for G in Genres:
  print(G)
  print("----------------------")
  print(final_data[final_data[G] == 1].groupby("Title")["Rating"].count().sort_values(ascending=False).head(5))
  print()
  print()
  print()


# Top 5 movie recommended as per age_Group based on ratings each age group provided
age_groups = final_data.Age.unique()
for age_ in age_groups:
  print(age_)
  print("------")
  print(final_data[final_data.Age == age_].groupby("Title")["Rating"].count().sort_values(ascending=False).head())
  print()
  print()
  print()





## Creating a user Movie average rating Matrix :
df_.columns

user_movie_rating_matrix = pd.pivot_table(df_,index = "UserID",
               columns = "Title",
               values = "Rating",
               aggfunc = "mean").fillna(0)
user_movie_rating_matrix.shape
user_movie_rating_matrix

## item item similarity(hamming distance) based recommendation :
m = movies[["MovieID","Title","Genres"]]
m["Genres"] = m["Genres"].str.split("|")
m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })

m = m.pivot_table(values="Title", index="MovieID", columns="Genres", aggfunc= np.size,).fillna(0)


def apply(x):
  if x >= 1:
    return 1
  else:
    return 0

m["Adventure"] = m["Adventure"].apply(apply)
m = m.astype(int)
m

def Hamming_distance(x1,x2):
  return np.sum(abs(x1-x2))

Ranks = []
Query = "1"
for candidate in m.index:
  if candidate == Query:
    continue
  Ranks.append([Query,candidate,Hamming_distance(m.loc[Query],m.loc[candidate])])

Ranks = pd.DataFrame(Ranks,columns=["Query","Candidate","Hamming_distance"])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Query', right_on='MovieID').rename(columns={'Title': 'query_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Candidate', right_on='MovieID').rename(columns={'Title': 'candidate_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.sort_values(by=['Query', 'Hamming_distance'])




Ranks.head(10)


def Hamming_distance(x1,x2):
  return np.sum(abs(x1-x2))

Ranks = []
Query = "1485"
for candidate in m.index:
  if candidate == Query:
    continue
  Ranks.append([Query,candidate,Hamming_distance(m.loc[Query],m.loc[candidate])])

Ranks = pd.DataFrame(Ranks,columns=["Query","Candidate","Hamming_distance"])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Query', right_on='MovieID').rename(columns={'Title': 'query_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Candidate', right_on='MovieID').rename(columns={'Title': 'candidate_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.sort_values(by=['Query', 'Hamming_distance'])




Ranks.head(10)





movies = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-movies.dat",encoding="ISO-8859-1")
ratings =pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-ratings.dat",encoding="ISO-8859-1")
users = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-users.dat",encoding="ISO-8859-1")
delimiter ="::"

users = users["UserID::Gender::Age::Occupation::Zip-code"].str.split(delimiter,expand = True)
users.columns = ["UserID","Gender","Age","Occupation","Zipcode"]

users["Age"].replace({"1": "Under 18","18": "18-24","25": "25-34",
                          "35": "35-44","45": "45-49","50": "50-55","56": "56+"},inplace=True)

users["Occupation"] = users["Occupation"].astype(int).replace({0: "other",1: "academic/educator",2: "artist",
                                                               3: "clerical/admin",4: "college/grad student",
                                             5: "customer service",6: "doctor/health care",7: "executive/managerial",
                                             8: "farmer" ,9: "homemaker",10: "K-12 student",11: "lawyer",
                                             12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",
                                             16: "self-employed",17: "technician/engineer",
                                             18: "tradesman/craftsman",19: "unemployed",20: "writer"},
                                            )

delimiter ="::"

ratings = ratings["UserID::MovieID::Rating::Timestamp"].str.split(delimiter,expand = True)
ratings.columns = ["UserID","MovieID","Rating","Timestamp"]


movies.drop(["Unnamed: 1","Unnamed: 2"],axis = 1,inplace=True)

delimiter ="::"

movies = movies["Movie ID::Title::Genres"].str.split(delimiter,expand = True)
movies.columns = ["MovieID","Title","Genres"]

movies.shape,ratings.shape,users.shape

movies["Release_year"] = movies["Title"].str.extract('^(.+)\s\(([0-9]*)\)$',expand = True)[1]
movies["Title"] = movies["Title"].str.split("(").apply(lambda x:x[0])

from datetime import datetime
ratings["Watch_Hour"] =ratings["Timestamp"].apply(lambda x:datetime.fromtimestamp(int(x)).hour)
ratings.drop(["Timestamp"],axis = 1,inplace=True)

df = users.merge(movies.merge(ratings,on="MovieID",how="outer"),on="UserID",how="outer")
df["Genres"] = df["Genres"].str.split("|")
df = df.explode('Genres')

df["Genres"] = df["Genres"].replace({"":"Other","Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama","Documenta":"Documentary",
                     "Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance","Animati":"Animation","Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi","Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action","Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller",
                        "Chi":"Children's","Ro":"Romance","F":"Fantasy","We":"Western","Documen":"Documentary","Music":"Musical","Children":"Children's" ,"Horr":"Horror",
                     "Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                    })
m = df.groupby(['MovieID','Genres'])['Title'].unique().str[0].unstack().reset_index().set_index('MovieID')
m = ~m.isna()
m = m.astype(int)




## Cosine Similarity :
## Item and User :  -Cosine similarity Matrix :
from sklearn.metrics.pairwise import cosine_similarity

Item_similarity = cosine_similarity(user_movie_rating_matrix.T)
Item_similarity
Item_similarty_matrix = pd.DataFrame(Item_similarity,
             index = user_movie_rating_matrix.columns,
             columns = user_movie_rating_matrix.columns)
Item_similarty_matrix


## User Based Similartiy :


User_similarity = cosine_similarity(user_movie_rating_matrix)
User_similarity.shape
User_similarity
User_similarity_matrix = pd.DataFrame(User_similarity,
             index = user_movie_rating_matrix.index,
             columns = user_movie_rating_matrix.index)
User_similarity_matrix




## Pearson Correlation


correlated_movie_matrix = m.T.corr()
correlated_movie_matrix
movies[movies.MovieID == "1"]["Title"][0]
movies[movies.Title.str.contains("Toy Story")].iloc[0].MovieID
def recommend_movie_based_on_correlation(movie):
    TITLE = movies[movies.Title.str.contains(movie)].iloc[0]["Title"]

    INDEX = movies[movies.Title.str.contains(movie)].iloc[0].MovieID

    print(TITLE)
    print(INDEX)

    print(movies[movies.MovieID.isin(correlated_movie_matrix[INDEX].sort_values(ascending=False).head(10).index.to_list())]["Title"])
recommend_movie_based_on_correlation("Toy Story")
recommend_movie_based_on_correlation("Shawshank")
recommend_movie_based_on_correlation("Titanic")
recommend_movie_based_on_correlation("Braveheart")



# k - Nearest Neighbours
from sklearn.neighbors import NearestNeighbors
kNN_model = NearestNeighbors(metric='cosine')
kNN_model.fit(user_movie_rating_matrix.T)
distances, indices = kNN_model.kneighbors(user_movie_rating_matrix.T, n_neighbors= 5)
result = pd.DataFrame(indices)
result
result.index = user_movie_rating_matrix.columns
result

 result.loc["Zero Effect "].to_list()
movies.MovieID = movies.MovieID.astype("int32")

movies[movies.MovieID.isin( result.loc["Zero Effect "].to_list())]







import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)

# from google.colab import drive
# drive.mount('/content/drive')
movies = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-movies.dat",encoding="ISO-8859-1")
ratings =pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-ratings.dat",encoding="ISO-8859-1")
users = pd.read_fwf("/notebooks/OTT_Data/ZEE-data/zee-users.dat",encoding="ISO-8859-1")
delimiter ="::"

users = users["UserID::Gender::Age::Occupation::Zip-code"].str.split(delimiter,expand = True)
users.columns = ["UserID","Gender","Age","Occupation","Zipcode"]

users["Age"].replace({"1": "Under 18","18": "18-24","25": "25-34",
                          "35": "35-44","45": "45-49","50": "50-55","56": "56+"},inplace=True)

users["Occupation"] = users["Occupation"].astype(int).replace({0: "other",1: "academic/educator",2: "artist",
                                                               3: "clerical/admin",4: "college/grad student",
                                             5: "customer service",6: "doctor/health care",7: "executive/managerial",
                                             8: "farmer" ,9: "homemaker",10: "K-12 student",11: "lawyer",
                                             12: "programmer",13: "retired",14: "sales/marketing",15: "scientist",
                                             16: "self-employed",17: "technician/engineer",
                                             18: "tradesman/craftsman",19: "unemployed",20: "writer"},
                                            )

delimiter ="::"

ratings = ratings["UserID::MovieID::Rating::Timestamp"].str.split(delimiter,expand = True)
ratings.columns = ["UserID","MovieID","Rating","Timestamp"]


movies.drop(["Unnamed: 1","Unnamed: 2"],axis = 1,inplace=True)



delimiter ="::"

movies = movies["Movie ID::Title::Genres"].str.split(delimiter,expand = True)
movies.columns = ["MovieID","Title","Genres"]


movies.shape,ratings.shape,users.shape

movies["Release_year"] = movies["Title"].str.extract('^(.+)\s\(([0-9]*)\)$',expand = True)[1]
movies["Title"] = movies["Title"].str.split("(").apply(lambda x:x[0])
from datetime import datetime
ratings["Watch_Hour"] =ratings["Timestamp"].apply(lambda x:datetime.fromtimestamp(int(x)).hour)
ratings.drop(["Timestamp"],axis = 1,inplace=True)
movies.shape,ratings.shape,users.shape

df = users.merge(movies.merge(ratings,on="MovieID",how="outer"),on="UserID",how="outer")
(df.isna().sum())/len(df)  * 100
data = df.copy()
data.dropna(inplace= True)
data
data.info()
data.nunique()
# 6040 unique UserID
# 7 different age groups
# 21 occupations
# 3493 different locations of users
# 3682 unique movies


# There are movies available in database , which were never been watched by any user before .
# Thats is the reason we have lots of NaN values in our final dataset.

data.shape
m = movies[["MovieID","Title","Genres"]]

m["Genres"] = m["Genres"].str.split("|")

m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"Horro":"Horror","Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi","Dr":"Drama",
                     "Documenta":"Documentary","Wester":"Western","Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance",
                     "Animati":"Animation","Childr":"Children's","Childre":"Children's","Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi",
                     "Roma":"Romance","A":"Adventure","Children":"Children's","Adventu":"Adventure","Adv":"Adventure",
                      "Wa":"War","Thrille"  :"Thriller","Com":"Comedy","Comed":"Comedy","Acti":"Action",
                        "Advent":"Adventure","Adventur":"Adventure","Thri":"Thriller","Chi":"Children's","Ro":"Romance",
                        "F":"Fantasy","We":"Western","Documen":"Documentary",
                        "Music":"Musical","Children":"Children's" ,"Horr":"Horror","Children'":"Children's","Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                     })
m


m = pd.crosstab(m["MovieID"],       m["Genres"])
m = pd.DataFrame(np.where(m>=1,1,0),index = m.index,columns=m.columns)

m
final_data = data.merge(m,on="MovieID",how="left").drop(["Genres"],axis = 1)
path = '/notebooks/OTT_Data/ZEE-data/final_data.csv '
final_data.to_csv(path)



## The movie with maximum no. of ratings is ___.


final_data.groupby("Title")["Rating"].count().reset_index().sort_values(by="Rating",ascending=False).head(10)
final_data.sample(2)
m = movies[["MovieID","Title","Genres"]]
m["Genres"] = m["Genres"].str.split("|")

m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"Horro":"Horror",
                     "Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi",
                     "Dr":"Drama",
                     "Documenta":"Documentary",
                     "Wester":"Western",
                     "Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance",
                     "Animati":"Animation",
                     "Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi",
                     "Roma":"Romance","A":"Adventure","Children":"Children's",
                     "Adventu":"Adventure",
                      "Adv":"Adventure",
                      "Wa":"War",
                      "Thrille"  :"Thriller"     ,
                      "Com"    :"Comedy"   ,
                      "Comed"    :"Comedy",
                      "Acti"   :"Action",
                        "Advent"   :"Adventure",
                        "Adventur"      :"Adventure",
                        "Thri":"Thriller",
                        "Chi":"Children's",
                        "Ro":"Romance",
                        "F":"Fantasy",
                        "We":"Western",
                        "Documen":"Documentary"       ,
                        "Music":"Musical"         ,
                        "Children":"Children's" ,
                        "Horr":"Horror"          ,
                     "Children'":"Children's",
                     "Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                     })

merged_data = ratings.merge(users,on="UserID",how="outer").merge(m,on="MovieID",how="outer")
merged_data.groupby("Genres")[["Title","UserID"]].nunique()
## Number of Movie TItles been raten as per Genre by each type of use Occupation :
Occupation_genre_count = merged_data.groupby(["Occupation","Genres"])["Title"].nunique().sort_values(ascending=False).reset_index()

Occupation_genre_count.pivot(index="Occupation",columns="Genres",values="Title")



pd.set_option("max_colwidth", None)

# Co-occurance | Frequency Based Recommender System (Apriory)
frame = data.groupby(["UserID","Title"])["Rating"].mean().unstack().reset_index().fillna(0).set_index('UserID')
frame = (frame > 0).astype(int)
frame.shape
pip install mlxtend

from mlxtend.frequent_patterns import apriori
frequent_itemsets_plus = apriori(frame, min_support=0.2,
                                 use_colnames=True).sort_values('support', ascending=False).reset_index(drop=True)


frequent_itemsets_plus['length'] = frequent_itemsets_plus['itemsets'].apply(lambda x: len(x))



frequent_itemsets_plus.shape
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets_plus, metric ="lift", min_threshold = 0.8)
rules.shape
rules.groupby(["antecedents"])["lift"].max().reset_index().merge(rules,on=["antecedents","lift"])
rules[rules["antecedents"] == rules.loc[4606]["antecedents"]].sort_values(by="lift",ascending=False).head(5)





### Item-Item Similarity Based Rec System

## Name the top 3 movies similar to ‘Liar Liar’ on the item-based approach.


movies[movies["Title"].str.contains("Liar Liar")]
m = movies[["MovieID","Title","Genres"]]

m = m.explode("Genres")
m["Genres"] = m["Genres"].replace({"Horro":"Horror",
                     "Sci-":"Sci-Fi","Sci":"Sci-Fi","Sci-F":"Sci-Fi",
                     "Dr":"Drama",
                     "Documenta":"Documentary",
                     "Wester":"Western",
                     "Fant":"Fantasy","Chil":"Children's","R":"Romance","D":"Drama","Rom":"Romance",
                     "Animati":"Animation",
                     "Childr":"Children's","Childre":"Children's",
                     "Fantas":"Fantasy","Come":"Comedy","Dram":"Drama","S":"Sci-Fi",
                     "Roma":"Romance","A":"Adventure","Children":"Children's",
                     "Adventu":"Adventure",
                      "Adv":"Adventure",
                      "Wa":"War",
                      "Thrille"  :"Thriller"     ,
                      "Com"        :"Comedy"   ,
                      "Comed"         :"Comedy",
                      "Acti"          :"Action",
                        "Advent"        :"Adventure",
                        "Adventur"      :"Adventure",
                        "Thri":"Thriller",
                        "Chi":"Children's",
                        "Ro":"Romance",
                        "F":"Fantasy",
                        "We":"Western",
                        "Documen":"Documentary"       ,
                        "Music":"Musical"         ,
                        "Children":"Children's" ,
                        "Horr":"Horror"          ,
                     "Children'":"Children's",
                     "Roman":"Romance","Docu":"Documentary","Th":"Thriller","Document":"Documentary"
                     })
m = pd.crosstab(m["MovieID"],m["Genres"])
m = pd.DataFrame(np.where(m>=1,1,0),index = m.index,columns=m.columns)
def Hamming_distance(x1,x2):
  return np.sum(abs(x1-x2))
Ranks = []
Query = "1485"
for candidate in m.index:
  if candidate == Query:
    continue
  Ranks.append([Query,candidate,Hamming_distance(m.loc[Query],m.loc[candidate])])
Ranks = pd.DataFrame(Ranks,columns=["Query","Candidate","Hamming_distance"])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Query', right_on='MovieID').rename(columns={'Title': 'query_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.merge(movies[['MovieID', 'Title']], left_on='Candidate', right_on='MovieID').rename(columns={'Title': 'candidate_tittle'}).drop(columns=['MovieID'])
Ranks = Ranks.sort_values(by=['Query', 'Hamming_distance'])



Ranks.head()





# Collaborative Filtering :

user_movie_ratings = ratings.pivot(index ="UserID",
              columns = "MovieID",
              values ="Rating").fillna(0)
user_movie_ratings.shape
# 6040 users # 3706 movies
ratings.shape, users.shape


rm_raw = ratings[['UserID', 'MovieID', 'Rating']].copy()
rm_raw.columns = ['UserId', 'ItemId', 'Rating']  # Lib requires specific column names

rm_raw.Rating = rm_raw.Rating.astype(int)
rm_raw.UserId = rm_raw.UserId.astype(int)
rm_raw.ItemId = rm_raw.ItemId.astype(int)
rm_raw.nunique()

!pip install cmfrec

from cmfrec import CMF

model = CMF(k=5, lambda_=0.1, user_bias=False, item_bias=False, verbose=False)
model.fit(rm_raw)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

rm_raw.shape,model.A_.shape,model.B_.shape
model.A_.shape,model.B_.T.shape

 model.topN(user=8, n=10)
 movies_to_recommend = model.topN(user=1, n=10)
 movies_to_recommend = movies_to_recommend[movies_to_recommend<3706]
 movies_to_recommend

 movies.MovieID = movies.MovieID.astype(int)
movies.loc[movies_to_recommend]

!pip install scikit-surprise




final_data= pd.read_csv("/notebooks/OTT_Data/ZEE-data/final_data.csv ")

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader

## The Reader class is used to parse a file containing ratings.It orders the data in format of (userid,title,rating) and even by considering the rating scale
reader = Reader(rating_scale=(0.5 , 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(final_data[["UserID","MovieID","Rating"]], reader) # loading the data as per the format
data
anti_set = data.build_full_trainset().build_anti_testset()
trainset, testset = train_test_split(data, test_size=.15) # Splitting the data
## User Based Collaborative Filtering :      

algo = KNNWithMeans(k = 50, sim_options={'name': 'cosine', 'user_based': True})

# K value represents the (max) number of neighbors to take into account for aggregation. Example for every item it gives 50 nearest ones.
# There are many similarity options to calculate the similarity between the neighbors. Here, we have used the cosine similarity.
# when user_based = True then it performs user based collaborative filtering

algo.fit(trainset) #fitting the train dataset
# run the trained model against the testset
test_pred = algo.test(testset)
test_pred[0]
# get RMSE on test set
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)

accuracy.mae(test_pred, verbose=True)
# we can query for specific predicions
uid = str(1)  # raw user id
iid = str(1)  # raw item id
pred = algo.predict(uid, iid, verbose=True)
# anti_pre = algo.test(anti_set)
# pred_df = pd.DataFrame(anti_pre).merge(movies , left_on = ['iid'], right_on = ['MovieID'])
# pred_df = pd.DataFrame(pred_df).merge(users , left_on = ['uid'], right_on = ['UserID'])
# Item Based Collaborative  Filtering :
# K value represents the (max) number of neighbors to take into account for aggregation. Example for every item it gives 50 nearest ones.
# There are many similarity options to calculate the similarity between the neighbors . Here, we have used the cosine similarity.
# when user_based = False then it performs item based collaborative filtering

algo_i = KNNWithMeans(k=30, sim_options={'name': 'cosine', 'user_based': False})
algo_i.fit(trainset)
test_pred = algo_i.test(testset)
test_pred[0]
# get RMSE on test set
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
# we can query for specific predicions
uid = str(196)  # raw user id
iid = str(303)  # raw item id

pred = algo_i.predict(uid, iid, verbose=True)
# final_data[final_data["MovieID"]=="984"]

# #tsr_inner_id = algo_i.trainset.to_inner_iid("1485") #Considering the movieId 1485 : Liar Liar

# tsr_inner_id = algo_i.trainset.to_inner_iid("984")

# tsr_neighbors = algo_i.get_neighbors(tsr_inner_id, k=5) #Getting the 5 nearest neighbors for movieId 1



# movies[movies.MovieID.isin([algo.trainset.to_raw_iid(inner_id)
                      #  for inner_id in tsr_neighbors])] #Displaying the 5 nearest neighbors to the Liar Liar

# Matrix Factorisation:
from surprise import SVD
from surprise.model_selection import cross_validate
svd = SVD() #Suprise library uses the SVD algorithm to perform the matrix factorisation where as other libraries uses ALS
cross_validate(svd, data, measures=['rmse','mae'], cv = 5 , return_train_measures=True,verbose=True)
##The dataset is divided into train and test and with 5 folds the rmse has been calculated

import pandas as pd
final_data= pd.read_csv("/notebooks/OTT_Data/ZEE-data/final_data.csv ")
from surprise import Dataset

from surprise import SVD

from surprise import Reader

## The Reader class is used to parse a file containing ratings.It orders the data in format of (userid,title,rating) and even by considering the rating scale
reader = Reader(rating_scale=(0.5 , 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(final_data[["UserID","MovieID","Rating"]], reader) # loading the data as per the format
svd = SVD(n_factors =10)
trainset = data.build_full_trainset()
svd.fit(trainset) ##Fitting the trainset with the help of svd
svd.pu.shape , svd.qi.shape #pu gives the embeddings of Users and qi gives the embeddings of Items.
#Storing all the movie titles in items
items = movies['Title'].unique()
##Considering the user '662'
test = [[662, iid, 4] for iid in items]
##Finding the user predictions(ratings) for all the movies
predictions = svd.test(test)
pred = pd.DataFrame(predictions)
a = pred.sort_values(by='est', ascending=False) ##Sorting the values based on the estimated predictions
a[0:10] ##TOP 10
testset = trainset.build_anti_testset()
predictions_svd = svd.test(testset) #Predicting for the test set
from surprise import accuracy
print('SVD - RMSE:', accuracy.rmse(predictions_svd, verbose=False))
print('SVD - MAE:', accuracy.mae(predictions_svd, verbose=False))


## Questions and Answers  :


1. Users of which age group have watched and rated the most number of movies?

    - age group 25-35


2. Users belonging to which profession have watched and rated the most movies?
    - College Graduate Students and Other category

3. Most of the users in our dataset who’ve rated the movies are Male. (T/F)
    - Male

4. Most of the movies present in our dataset were released in which decade?
    - 90s



5. The movie with maximum no. of ratings is ___.
    - American Beauty

6. Name the top 3 movies similar to ‘Liar Liar’ on the item-based approach.

    - The Associate
    - Ed's Next Move
    - Bottle Rocket
    - Mr. Wrong
    - Cool Runnings
    - Happy Gilmore
    - That Thing You Do!





7. On the basis of approach, Collaborative Filtering methods can be classified into ___-based and ___-based.
      
    - Memory based and Model based
    


8. Pearson Correlation ranges between ___ to ___ whereas, Cosine Similarity belongs to the interval between ___ to ___.
    - Pearson Correlation ranges between -1 to +1
    - Cosine Similarity belongs to the interval between -1 to 1

    - similarity of 1 means that the vectors are identical,
    - a similarity of -1 means that the vectors are dissimilar,
    - and a similarity of 0 means that the vectors are not similar.


9. Mention the RMSE and MAPE that you got while evaluating the Matrix Factorization model.
    - Item-based Model :
    - RMSE: 0.8926
    - User-based Model :
    - RMSE: 0.9345



10. Give the sparse ‘row’ matrix representation for the following dense matrix -

    - [[1 0],[3 7]]

            ans  :
                    [1 3 7]
                    [0 0 1]
                    [0 1 3]
            
from scipy.sparse import csr_matrix

dense_matrix = [[1,0],
                [3,7]]
sparse_matrix = csr_matrix(dense_matrix)
print(sparse_matrix.data)
print(sparse_matrix.indices)
print(sparse_matrix.indptr)





