# 🎬 Movie Recommendation System using Machine Learning

A **Content-Based Movie Recommendation System** built using **Python and Machine Learning**.
The system recommends movies similar to a movie entered by the user by analyzing movie features such as **genres, keywords, tagline, cast, and director**.

---

## 📌 Project Overview

Finding a good movie to watch can sometimes be difficult when there are thousands of movies available.

This project solves this problem by building a recommendation system that takes the user's favorite movie as input and recommends movies that are most similar to it.

The project uses **Natural Language Processing (NLP)** techniques and **Cosine Similarity** to calculate the similarity between movies.

---

## 🚀 Features

* 🎥 Takes a movie name as user input
* 🔍 Finds the closest matching movie name
* 🧹 Handles missing values in movie features
* 📝 Combines important movie information
* 🔢 Converts text data into numerical feature vectors
* 📊 Uses TF-IDF Vectorization
* 📐 Calculates similarity using Cosine Similarity
* ⭐ Recommends movies similar to the selected movie

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Difflib**
* **Jupyter Notebook / Google Colab**

---

## 📂 Dataset

The project uses a movie dataset containing **4,803 movies and 24 columns**.

Some important features in the dataset include:

* Genres
* Keywords
* Tagline
* Cast
* Director
* Title
* Overview
* Popularity
* Vote Average
* Vote Count

The project selects the following five features for generating recommendations:

```text
genres
keywords
tagline
cast
director
```

---

## 🔄 How the System Works

The recommendation system follows these steps:

```text
Movie Dataset
      ↓
Data Preprocessing
      ↓
Select Important Features
      ↓
Handle Missing Values
      ↓
Combine Movie Features
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find User's Movie
      ↓
Find Similar Movies
      ↓
Display Recommendations
```

---

## 🧠 Methodology

### 1. Data Collection

The movie dataset is loaded into a Pandas DataFrame.

```python
movies_data = pd.read_csv('/content/movies.csv')
```

The dataset contains 4,803 rows and 24 columns.

---

### 2. Selecting Important Features

The following features are selected because they provide useful information about movie similarity:

```python
selected_features = [
    'genres',
    'keywords',
    'tagline',
    'cast',
    'director'
]
```

---

### 3. Handling Missing Values

Missing values in the selected features are replaced with empty strings.

```python
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
```

This prevents missing values from causing problems while combining the features.

---

### 4. Combining Features

The selected movie features are combined into a single text representation.

```python
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)
```

This combined text is then used for creating feature vectors.

---

### 5. TF-IDF Vectorization

The combined movie information is converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)
```

TF-IDF helps represent the importance of words in the movie information.

---

### 6. Cosine Similarity

Cosine Similarity is used to measure how similar two movie feature vectors are.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(feature_vectors)
```

The resulting similarity matrix has the shape:

```text
(4803, 4803)
```

Each value represents the similarity between two movies.

---

### 7. Finding the User's Movie

The user enters their favorite movie:

```python
movie_name = input('Enter your favourite movie name: ')
```

For example:

```text
iron man
```

The system uses `difflib.get_close_matches()` to find the closest movie title available in the dataset.

```python
find_close_match = difflib.get_close_matches(
    movie_name,
    list_of_all_titles
)
```

For `"iron man"`, possible matches include:

```text
Iron Man
Iron Man 3
Iron Man 2
```

The closest match is selected.

---

### 8. Finding Similar Movies

The index of the selected movie is obtained and its similarity scores are retrieved.

```python
index_of_the_movie = movies_data[
    movies_data.title == close_match
]['index'].values[0]

similarity_score = list(
    enumerate(similarity[index_of_the_movie])
)
```

The movies are then ranked based on their similarity scores.

---

## 🎯 Example

### Input

```text
Enter your favourite movie name: iron man
```

### Closest Match

```text
Iron Man
```

The system then uses the similarity scores of **Iron Man** to identify movies with similar characteristics.

---

## 📁 Project Structure

```text
Movie-Recommendation-System-using-ML-with-Python/
│
├── Movie_Recommendation_System_using_ML_with_Python.ipynb
├── movies.csv
└── README.md
```

> If your GitHub repository does not contain `movies.csv`, remove it from the structure above and add instructions explaining where the dataset should be obtained.

---

## ▶️ How to Run the Project

### Option 1: Google Colab

Open the notebook directly in Google Colab:

[Open in Google Colab](https://colab.research.google.com/github/parameshgorentla30-ctrl/Movie-Recommendation-System-using-ML-with-Python/blob/main/Movie_Recommendation_System_using_ML_with_Python.ipynb)

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/parameshgorentla30-ctrl/Movie-Recommendation-System-using-ML-with-Python.git
```

Go to the project folder:

```bash
cd Movie-Recommendation-System-using-ML-with-Python
```

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Movie_Recommendation_System_using_ML_with_Python.ipynb
```

---

## 📦 Required Libraries

```python
import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

---

## 💡 Machine Learning Concepts Used

This project demonstrates the following concepts:

* Content-Based Recommendation
* Natural Language Processing
* Feature Extraction
* TF-IDF Vectorization
* Cosine Similarity
* Data Preprocessing
* Missing Value Handling
* Text Similarity

---

## 📊 Project Highlights

| Component            | Details           |
| -------------------- | ----------------- |
| Dataset Size         | 4,803 movies      |
| Dataset Columns      | 24                |
| Recommendation Type  | Content-Based     |
| Vectorization        | TF-IDF            |
| Similarity Algorithm | Cosine Similarity |
| Input Matching       | Difflib           |
| Language             | Python            |

---

## 🔮 Future Improvements

The project can be further improved by adding:

* 🎞️ Movie posters and thumbnails
* ⭐ Movie rating information
* 🎭 Genre-based filtering
* 👤 User-based recommendations
* 🌐 Web application using Flask or Streamlit
* 🔎 Better movie search
* 📈 Recommendation ranking
* 🎬 Movie details such as release year, rating, and overview

---

## 🎓 Learning Outcomes

Through this project, I learned how to:

* Work with real-world datasets
* Perform data preprocessing using Pandas
* Handle missing values
* Select relevant features
* Convert text into numerical vectors
* Apply TF-IDF Vectorization
* Calculate Cosine Similarity
* Build a basic recommendation system using Machine Learning

---

## 👨‍💻 Author

**Paramesh Gorentla**

GitHub:
https://github.com/parameshgorentla30-ctrl

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub!
