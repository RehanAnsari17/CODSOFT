# Movie Genre Classification

## 📌 Overview
This project develops a **machine learning pipeline** to classify movies into genres using their textual metadata—primarily movie titles and descriptions.  
It demonstrates how text data can be leveraged for multi-class classification tasks in the entertainment domain.

***

## 📂 Dataset
Dataset Link - Kaggle[https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb}
The dataset includes:
- **Movie ID** — Unique identifier for each film
- **Title** — Name of the movie
- **Genre** — The target variable (e.g., drama, thriller, comedy, documentary, adult)
- **Description** — Brief summary of the movie plot or theme

The data appears to be stored in `.txt` files with custom separators and includes both training and testing sets.

***

## ⚙️ Project Workflow

1. **Import Dependencies**
   - Main libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`
   - Machine learning: `scikit-learn` (`TfidfVectorizer`, `LabelEncoder`, `LinearSVC`, etc.)

2. **Load Data**
   - Uses `pandas.read_csv()` with a custom separator to read movie metadata files.

3. **Exploratory Data Analysis (EDA)**
   - Preview of dataset samples.
   - Inspection of class (genre) distribution.

4. **Feature Engineering & Text Processing**
   - **TfidfVectorizer** converts movie descriptions to numerical features for model training.
   - **LabelEncoder** encodes genre labels for classification.

5. **Train/Test Split**
   - Dataset is split into training and testing portions for validation.

6. **Model Training**
   - Core model: **LinearSVC** (Support Vector Classifier)
   - Alternative experiments (mentioned): Naive Bayes, Logistic Regression.
   
7. **Model Evaluation**
   - Metrics: **Accuracy Score**, **Classification Report** (Precision, Recall, F1-score per genre)
   - Visualization of genre prediction performance (optional with seaborn/matplotlib).

***

## 📊 Sample Preprocessed Columns

| Column Name     | Description                                        |
|-----------------|----------------------------------------------------|
| `ID`            | Unique movie identifier                            |
| `TITLE`         | Movie title                                        |
| `GENRE`         | True genre label                                   |
| `DESCRIPTION`   | Movie description text                             |
| `TF-IDF features` | Vectorized numeric features extracted from description |
| `Encoded Genre` | Numeric code for each genre                        |

***

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/movie-genre-classification.git
cd movie-genre-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or manually install: `pandas numpy seaborn matplotlib scikit-learn`)*

### 3. Add the Dataset
- Place `train_data.txt` and (`test_data.txt` or similar) files inside a `data/` folder.

### 4. Run the Notebook
```bash
jupyter notebook Movie_Genre_Classification.ipynb
```

***

## 📈 Results
- The **LinearSVC classifier** provides competitive accuracy for genre prediction using only textual features.
- **Classification report** details the precision, recall, and F1-score for each genre, indicating model strengths and weaknesses.

***

## 🔮 Future Improvements
- Try **Random Forest** or **Gradient Boosting** models for non-linear feature interactions.
- Explore **deep learning approaches** like LSTMs or Transformers for richer text modeling.
- Incorporate additional metadata (e.g., cast, director, year).
- Perform **hyperparameter optimization** for top accuracy.


## 🙌 Acknowledgments
- Thanks to public movie datasets for enabling genre prediction research.
- **CodSoft** for giving me this task.

***


[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/83245422/b011f71a-6531-4406-af94-9558d641897e/Movie_Genre_Classification.ipynb
