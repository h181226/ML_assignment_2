# careful - some erorrs and stuff still inside


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/.kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Import nessecary libraries and prepare some settings for plotting
import IPython
import numpy as np
import pandas as pd
import streamlit as st
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from zlib import crc32
import tarfile
import urllib.request

# To plot pretty figures
#matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
p_root = "." #project root
c_id = "end_to_end_project" #chapter ID
i_path = os.path.join(p_root, "images", c_id) #image path
os.makedirs(i_path, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(i_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

train = pd.read_csv('~/.kaggle/input/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('~/.kaggle/input/tmdb-box-office-prediction/test.csv')
sample = pd.read_csv('~/.kaggle/input/tmdb-box-office-prediction/sample_submission.csv')

train.head()
# countable features: budget, popularity, runtime, revenue

test.head()
# countable features: budget, popularity, runtime

train.info()
# values missing in belongs_to, genres, homepage, overview, poster_path, production_companies, production_countries, runtime, spoken_languages, tagline, keywords, cast and crew
# important to fill: runtime
# not really possible: genres, homepage, overview, poster_path, production_companies, production_countries, spoken_languages, tagline, keywords, cast and crew

test.info()

train.describe()

test.describe()

len(train)

len(test)

train.head()
# original_language, genres, keywords? change

#matplotlib inline
import matplotlib.pyplot as plt
train.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
st.pyplot()


#matplotlib inline
test.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots_test")
st.pyplot()

train["pop_per_day"] = train["popularity"]/train["runtime"]

train.plot(kind="scatter", x="revenue", y="pop_per_day", alpha=0.1)
save_fig("better_visualization_plot")
#no

from pandas.plotting import scatter_matrix

attributes = ["budget", "runtime", "popularity", "revenue"]
scatter_matrix(train[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")


corr_matrix = train.corr()
corr_matrix["revenue"].sort_values(ascending=False)
# Still not very useful

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
#
# train_num = train["runtime"]
# train_num = np.array(train_num.values.tolist())
# train_num = train_num.reshape(-1,1)
# #train_num = train_num.astype(float)
# imputer.fit(train_num)
#
# print(train_num)
#
# X = imputer.transform(train_num)
#
# train_tr = pd.DataFrame(X, columns=train_num,
#                           index=train("runtime"))
#
# train_tr = pd.DataFrame(X, columns=train_num.columns,index=train.index)
