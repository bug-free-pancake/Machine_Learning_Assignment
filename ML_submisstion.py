import pandas as pd
import numpy as np
import json
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordsegment import load, segment
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cluster import KMeans
from scipy.stats import uniform, randint


# Load the data
train = pd.DataFrame.from_records(json.load(open('train.json')))
test = pd.DataFrame.from_records(json.load(open('test.json')))


# Drop the 'id' column that does not contribute to finding patterns 
train.drop(columns=['id'], inplace=True)


# Preprocess text data 
load()
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # https://tilburgsciencehub.com/topics/manage-manipulate/manipulate-clean/textual/text-prep-python/

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  

    # Reduce excessive character repetition
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  

    # Remove stand-alone numbers 
    text = re.sub(r'\b\d+\b', '', text)  

    # Replace repetitive 'null' patterns with 'unknown'
    text = re.sub(r'(null)+', 'unknown', text, flags=re.IGNORECASE)  

    # Normalize extra spaces, trim and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()   

    # Separate words linked together 
    text = ' '.join(word 
                    if len(word) <= 15 
                    else ' '.join(segment(word)) 
                    for word in text.split())  

    # Tokenize and lemmatize the text
    tokens = [lemmatizer.lemmatize(word) 
              for word in word_tokenize(text) 
              if len(word) > 2]

    return ' '.join(tokens) if tokens else 'unknown'


# Clean text in 'abstract', 'title' and 'venue' columns in training set
train['abstract'] = train['abstract'].apply(clean_text)
train['title'] = train['title'].apply(clean_text)
train['venue'] = train['venue'].apply(clean_text)

# Clean text in 'abstract', 'title' and 'venue' columns in test set
test['abstract'] = test['abstract'].apply(clean_text)
test['title'] = test['title'].apply(clean_text)
test['venue'] = test['venue'].apply(clean_text)


# Count empty strings in each column
# empty_strings_per_column = train.isin(['']).sum()
# print(empty_strings_per_column)


# Replace empty strings in 'venue' column with 'unknown' 
train['venue'] = train['venue'].replace('', np.nan).fillna('unknown')
test['venue'] = test['venue'].replace('', np.nan).fillna('unknown')


# Create a binary feature indicating whether the venue is 'unknown'
train['has_venue'] = train['venue'].apply(
    lambda x: 0 if x == 'unknown' else 1)
test['has_venue'] = test['venue'].apply(
    lambda x: 0 if x == 'unknown' else 1)


# Add a new column with the name of the principle author for each publication
train['first_author'] = train['authors'].apply(
    lambda x: x.split(',')[0].strip())
test['first_author'] = test['authors'].apply(
    lambda x: x.split(',')[0].strip())


# Transform ‘years’ into how many years each article has been published 
train['year'] = 2024 - train['year']
train = train.rename(columns={'year':'age'})
test['year'] = 2024 - test['year']
test = test.rename(columns={'year':'age'})


# Define a function counting the number of words in a text
def text_length(text):
    new_text = text.split()
    return len(new_text)  


# Add a new column calculating the length of each abstract
train['abstract_length'] = train['abstract'].apply(text_length)
test['abstract_length'] = test['abstract'].apply(text_length)


# Add a new column caculating the length of each title
train['title_length'] = train['title'].apply(text_length)
test['title_length'] = test['title'].apply(text_length)


# Add a new column counting the number of authors for each publication
train['num_authors'] = train['authors'].apply(
    lambda x: len(x.split(',')))
test['num_authors'] = test['authors'].apply(
    lambda x: len(x.split(',')))


# Add a new column counting the number of references for each publication
train['num_references'] = train['references'].apply(
    lambda x: len(x))
test['num_references'] = test['references'].apply(
    lambda x: len(x))


# Create a binary feature indicating whether the publication has references
train['has_references'] = train['num_references'].apply(
    lambda x: 0 if x == 0 else 1)
test['has_references'] = test['num_references'].apply(
    lambda x: 0 if x == 0 else 1)


# Identify frequently repeated standard abstract templates
template_abstracts = train['abstract'].value_counts().loc[
    lambda x: (x > 2) & (x.index != 'unknown')].index
template_abstracts_test = test['abstract'].value_counts().loc[
    lambda x: (x > 2) & (x.index != 'unknown')].index


# Replace template abstracts with a placeholder
train['abstract'] = train['abstract'].apply(
    lambda x: 'template' if x in template_abstracts else x)
test['abstract'] = test['abstract'].apply(
    lambda x: 'template' if x in template_abstracts_test else x)


# Create a binary feature indicating whether the abstract is using a template
train['is_template'] = train['abstract'].apply(
    lambda x: 1 if x == 'template' else 0)
test['is_template'] = test['abstract'].apply(
    lambda x: 1 if x == 'template' else 0)


# Inspect data distribution of numerical features 
# (can observe high positive skewness and presence of outliers)

# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))

# axes[0, 0].boxplot(train['num_authors'])
# axes[0, 0].set_title('Num Authors')

# axes[0, 1].boxplot(train['num_references'])
# axes[0, 1].set_title('Num References')

# axes[1, 0].boxplot(train['age'])
# axes[1, 0].set_title('Age')

# axes[1, 1].boxplot(train['n_citation'])
# axes[1, 1].set_title('N Citation')

# axes[2, 0].boxplot(train['abstract_length'])
# axes[2, 0].set_title('abstract_length')

# axes[2, 1].boxplot(train['title_length'])
# axes[2, 1].set_title('title_length')

# plt.tight_layout()
# plt.show()


# Replace the outliers with the nearest non-outlying value for numerical features

# Calculate outer fence for 'num_authors'
Q1_authors = train['num_authors'].quantile(0.25)
Q3_authors = train['num_authors'].quantile(0.75)
IQR_authors = Q3_authors - Q1_authors
upper_bound_authors = Q3_authors + 3 * IQR_authors

# Cap the outliers within the outer fence
train['num_authors'] = train['num_authors'].clip(
    upper=upper_bound_authors)


# Calculate the outer fence for 'num_references'
Q1_references = train['num_references'].quantile(0.25)
Q3_references = train['num_references'].quantile(0.75)
IQR_references = Q3_references - Q1_references
upper_bound_references = Q3_references + 3 * IQR_references

# Cap the outliers within the outer fence
train['num_references'] = train['num_references'].clip(
    upper=upper_bound_references)


# Calculate the outer fence for 'abstract_length'
Q1_abstract = train['abstract_length'].quantile(0.25)
Q3_abstract = train['abstract_length'].quantile(0.75)
IQR_abstract = Q3_abstract - Q1_abstract
upper_bound_abstract = Q3_abstract + 3 * IQR_abstract

# Cap the outliers within the outer fence
train['abstract_length'] = train['abstract_length'].clip(
    upper=upper_bound_abstract)


# Calculate the outer fence for 'title_length'
Q1_title = train['title_length'].quantile(0.25)
Q3_title = train['title_length'].quantile(0.75)
IQR_title = Q3_title - Q1_title
upper_bound_title = Q3_title + 3 * IQR_title

# Cap the outliers within the outer fence
train['title_length'] = train['title_length'].clip(
    upper=upper_bound_title)


# Calculate the outer fence for 'age'
Q1_age = train['age'].quantile(0.25)
Q3_age = train['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
upper_bound_age = Q3_age + 3 * IQR_age

# Cap the outliers within the outer fence
train['age'] = train['age'].clip(upper=upper_bound_age)


# Check the target statistics
# train.n_citation.describe().astype(int)


# Define a function to apply logrithmic transformation
def apply_log_transform(df, columns):
    
    for col in columns:
        df[f'log_{col}'] = np.log1p(df[col])
    return df


# Log transform integer count data
numerical_features_to_transform = [
    'age', 
    'num_authors', 
    'num_references', 
    'abstract_length', 
    'title_length']
train = apply_log_transform(train, 
                            numerical_features_to_transform + ['n_citation'])
test = apply_log_transform(test, 
                           numerical_features_to_transform) 


# Add interaction terms to training set 
features_to_add_interactions = ['log_age', 
                                'log_num_authors', 
                                'log_num_references', 
                                'log_abstract_length', 
                                'log_title_length']
X = train[features_to_add_interactions]
poly = PolynomialFeatures(degree=2, 
                          interaction_only=True, 
                          include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)
poly_feature_names = poly.get_feature_names_out(
    features_to_add_interactions)

# Concatenate the interaction terms to the original training set
X_poly_df = pd.DataFrame(X_poly, 
                         columns=poly_feature_names, 
                         index=train.index).drop(
                             features_to_add_interactions, 
                             axis=1)
train = pd.concat([train, X_poly_df], axis=1)


# Add interaction terms to the test set 
X_test = test[features_to_add_interactions]
poly_test = PolynomialFeatures(degree=2, 
                               interaction_only=True, 
                               include_bias=False)
poly_test.fit(X_test)
X_test_poly = poly_test.transform(X_test)
poly_test_feature_names = poly_test.get_feature_names_out(
    features_to_add_interactions)

# Concatenate the interaction terms to the original test set
X_test_poly_df = pd.DataFrame(X_test_poly, 
                              columns=poly_test_feature_names, 
                              index=test.index).drop(
                                  features_to_add_interactions, 
                                  axis=1)
test = pd.concat([test, X_test_poly_df], axis=1)


# Standardize all transformed numerical features

# Select the features to standardize
features_to_scale = [
    'log_age', 'log_num_authors', 
    'log_num_references', 
    'log_abstract_length', 
    'log_title_length', 
    'log_age log_num_authors', 
    'log_age log_num_references', 
    'log_age log_abstract_length', 
    'log_age log_title_length', 
    'log_num_authors log_num_references', 
    'log_num_authors log_abstract_length', 
    'log_num_authors log_title_length', 
    'log_num_references log_abstract_length', 
    'log_num_references log_title_length', 
    'log_abstract_length log_title_length'
]

# Initialize scaler
scaler = StandardScaler()

# Scale the selected features
train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
test[features_to_scale] = scaler.transform(test[features_to_scale])


# Transform abstract length from continuous variable into bins

# Calculate the quantiles on the training set
quantiles = pd.qcut(
    train['abstract_length'], 
    q=4, 
    retbins=True
)[1]  

# Apply the quantiles to the training set
train['abstract_length_binned'] = pd.cut(
    train['abstract_length'], 
    bins=quantiles, 
    labels=['very short', 'short', 'medium', 'long'], 
    include_lowest=True)

# Apply the same quantiles to the test set
test['abstract_length_binned'] = pd.cut(
    test['abstract_length'], 
    bins=quantiles, 
    labels=['very short', 'short', 'medium', 'long'], 
    include_lowest=True)


# Compute the correlation matrix for all numeric columns
# correlation_matrix = train.corr(numeric_only=True)
# log_features_correlation = correlation_matrix.loc[features_to_scale, 
#                                                   features_to_scale]
# log_features_correlation

# Single out the list of correlations with the target variable
# log_features_with_target = correlation_matrix.loc[features_to_scale, 
#                                                  'log_n_citation'].sort_values(ascending=False)

# print('\nCorrelations with Target Variable:')
# print(log_features_with_target)

# Drop variable with near-zero correlation with the target
columns_weak_correlation = ['log_num_authors log_title_length', 
                            'log_num_authors', 
                            'log_abstract_length log_title_length', 
                            'log_num_authors log_abstract_length']
train.drop(columns=columns_weak_correlation, inplace=True)
test.drop(columns=columns_weak_correlation, inplace=True)


# Plot the two features with the highest correlations with the target variable
# (The dense clusters and wide spread suggest potential non-linear relationships)

# Scatterplot for 'log_age log_abstract_length' interaction term and 'log_n_citation'
# sns.scatterplot(x = train['log_age log_abstract_length'], y = train['log_n_citation'])
# plt.title('log_age log_abstract_length')
# plt.show()

# Scatterplot for 'log_age' and 'log_n_citation'
# sns.scatterplot(x = train['log_age'], y = train['log_n_citation'])
# plt.title('log_age')
# plt.show()


# Define a function to group 'venues' into common publication types

def categorize_venue(venue):
    '''
    filter out some common publication venue types present in the data
    '''
    if venue == 'unknown':
        return 'unknown'
    elif 'journal' in venue:
        return 'journal'
    elif 'conference' in venue:
        return 'conference'
    elif 'symposium' in venue:
        return 'symposium'
    elif 'lecture' in venue:
        return 'lecture'
    elif 'magazine' in venue:
        return 'magazine'
    elif 'forum' in venue:
        return 'forum'
    elif 'report' in venue:
        return 'report'
    elif 'thesis' in venue:
        return 'thesis'
    elif 'workshop' in venue:
        return 'workshop'
    elif 'summit' in venue:
        return 'summit'
    elif 'meeting' in venue:
        return 'meeting'
    elif 'review' in venue:
        return 'review'
    elif 'method' in venue or 'methodology' in venue:
        return 'method'
    elif 'paper' in venue:
        return 'paper'
    elif 'open source' in venue:
        return 'open_source'
    elif 'letter' in venue:
        return 'letter'
    else:
        return 'other'


# Put the venues into common publication categories
train['venue_categories'] = train['venue'].apply(categorize_venue)
test['venue_categories'] = test['venue'].apply(categorize_venue)

# Extract the venues that do not indicate types by its name
other_venues = train[train['venue_categories'] == 'other']['venue']
other_venues_test = test[test['venue_categories'] == 'other']['venue']

# Vectorize the venue names to further categorize them
vectorizer = TfidfVectorizer(max_features=50, 
                             ngram_range=(1, 2), 
                             stop_words='english', 
                             min_df=1, 
                             max_df=1.0) 
venue_vectors = vectorizer.fit_transform(other_venues)

# Reduce dimensions 
svd = TruncatedSVD(n_components=30, random_state=123)
reduced_vectors = svd.fit_transform(venue_vectors)

# Cluster the venues
kmeans = KMeans(n_clusters=16, random_state=123)
clusters = kmeans.fit_predict(reduced_vectors)


# Create a new column to hold the subdivided categories
train.loc[train['venue_categories'] == 'other', 'venue_grouped'] = clusters

# Check the venues in each cluster
# for cluster_id in range(16):
#    print(f'Cluster {cluster_id}')
#    print(other_venues[clusters == cluster_id].head(10))  
#    print('\n')

# Assign meaningful names to the clusters 
cluster_mapping = {
    0: 'pattern_recognition',
    1: 'iee',
    2: 'others',
    3: 'information',
    4: 'computer_science', 
    5: 'computing',
    6: 'communication',
    7: 'application',
    8: 'software',
    9: 'reliability',
    10: 'computational',
    11: 'info_tech',
    12: 'discrete_mathematics',
    13: 'computation',
    14: 'data',
    15: 'design'
}

# Replace the numerical representation of clusters in the column with the names
train['venue_grouped'] = train['venue_grouped'].map(cluster_mapping)

# Merge the subdivided categories with the existing grouping
train['venue_grouped'] = train['venue_grouped'].fillna(train['venue_categories'])

# Drop the previous grouping column
train.drop(columns=['venue_categories'], inplace=True)


# Save vectorizer and KMeans models from the training set
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(svd, 'svd_model.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

# Load saved models
vectorizer = joblib.load('vectorizer.pkl')
svd = joblib.load('svd_model.pkl')
kmeans = joblib.load('kmeans_model.pkl')


# Transform the venue column in the test set using the vectorizer and KMeans models
test_venue_vectors = vectorizer.transform(other_venues_test)
test_reduced_vectors = svd.transform(test_venue_vectors)
test_clusters = kmeans.predict(test_reduced_vectors)

# Create a new column to hold the subdivided categories in the test set
test.loc[test['venue_categories'] == 'other', 'venue_grouped'] = test_clusters

# Map the cluster names in the column 
test['venue_grouped'] = test['venue_grouped'].map(cluster_mapping)

# Merge the subdivided categories with the existing grouping
test['venue_grouped'] = test['venue_grouped'].fillna(test['venue_categories'])

# Drop the previous grouping column
test.drop(columns=['venue_categories'], inplace=True)

# Extract training and test data from the processed data 
train_set = train.drop(
    columns=numerical_features_to_transform+['references'], 
    inplace=False)
test_set = test.drop(
    columns=numerical_features_to_transform+['references'], 
    inplace=False)


# In practice, worked on a subsample due to computational constraint
subsample = train_set.sample(frac=0.05, random_state=123)

# Extract feature columns 
X = subsample.drop(columns=['n_citation', 'log_n_citation'])

# Extract target variable
y = subsample['log_n_citation']

# Split the data into training and test sets 
# (hyperparameters tuned with random search instead of an explicit validation set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Define LDA pipeline for abstract topics
abstract_topic_pipeline = Pipeline([
    ("vectorizer", CountVectorizer(
        analyzer="word",
        token_pattern=r"\b\w{2,}\b",
        stop_words="english",
        max_features=105,
        ngram_range=(1, 1),
        min_df=1, max_df=0.8
    )),
    ("lda", LatentDirichletAllocation(
        n_components=22,  
        max_iter=10,
        random_state=123,
        doc_topic_prior=0.1,
        learning_decay=0.6,
        topic_word_prior=0.1,
        learning_method="online"
    ))
])


# Define LDA pipeline for title topics 
title_topic_pipeline = Pipeline([
    ("vectorizer", CountVectorizer(
        analyzer="word",
        token_pattern=r"\b\w{2,}\b",
        stop_words="english",
        max_features=53,
        ngram_range=(1, 1),
        min_df=3, max_df=0.7
    )),
    ("lda", LatentDirichletAllocation(
        n_components=33,  
        max_iter=10,
        random_state=123,
        doc_topic_prior=0.1,
        topic_word_prior=None,
        learning_decay=0.7,
        learning_method="online"
    ))
])

# 

# Define a label encoder for 'venue_grouped'
label_encoder = LabelEncoder()
label_encoder.fit(train['venue_grouped'])

def venue_grouped_transformer_func(X):
    return label_encoder.transform(X).reshape(-1, 1)

# Wrap it in FunctionTransformer
venue_grouped_transformer = FunctionTransformer(venue_grouped_transformer_func, validate=False)


# Define a ColumnTransformer
featurizer = ColumnTransformer(
        transformers=[
            
            # TF-IDF for 'abstract'
            ('abstract', 
             TfidfVectorizer(
                analyzer='word', 
                token_pattern=r'\b\w{2,}\b', 
                ngram_range=(1,2), 
                stop_words='english', 
                max_features=115, 
                min_df=1, max_df=0.8), 
             'abstract'),

            # TF-IDF for 'authors'
            ('authors', 
             TfidfVectorizer(
                token_pattern=None, 
                tokenizer=lambda x: x.split(','), 
                ngram_range=(1, 1), 
                max_features=61, 
                min_df=3, max_df=0.8), 
             'authors'),

            # TF-IDF for 'title'
            ('title', 
             TfidfVectorizer(
                analyzer='word', 
                token_pattern=r'\b\w{2,}\b', 
                ngram_range=(1,2), 
                stop_words='english', 
                max_features=74, 
                min_df=3, max_df=0.8), 
             'title'),

            # TF-IDF for 'venue'
            ('venue', 
             TfidfVectorizer(
                analyzer='word', 
                token_pattern=r'\b\w{2,}\b', 
                ngram_range=(1,2), 
                stop_words='english', 
                max_features=86, 
                min_df=1, max_df=0.8), 
             'venue'),

            # Extract topics from 'abstract' 
            ('abstract_lda', 
             abstract_topic_pipeline, 
             'abstract'),

            # Extract topics from 'title'
            # ('title_lda', 
            # title_topic_pipeline, 
            # 'title'),  # dropped for decrease in performance

            # Frequency encoding and scaling for 'venue_grouped'
            ('venue_grouped', 
             Pipeline([
                ('label_enc', 
                 venue_grouped_transformer),
                ('scaler', 
                 StandardScaler())]), 
             'venue_grouped'),

            # One-hot encoding for 'first_author'
            ('first_author', 
             OneHotEncoder(
                 handle_unknown='ignore', 
                 max_categories=15), 
                 ['first_author']),

            # One-hot encoding for 'venue_transformed'
            ('venue_transformed', 
             OneHotEncoder(
                handle_unknown='ignore', 
                max_categories=15),
                ['venue']), 

            # One-hot encoding for 'abstract_length_binned'
            ('abstract_length_binned', 
             OneHotEncoder(),
             ['abstract_length_binned']),

            ('drop_features', 
             'drop', 
             ['abstract_length_binned'])  # drop features that decreased performance
        ],
    sparse_threshold=0.0,
    remainder='passthrough')


# Initiate the models
# dummy = make_pipeline(
#     featurizer, 
#     DummyRegressor())

# GXboost = make_pipeline(
#     featurizer, 
#     GradientBoostingRegressor(
#        learning_rate=0.05, 
#        max_depth=6, 
#        max_features='sqrt', 
#        min_samples_leaf=20, 
#        min_samples_split=15, 
#        n_estimators=300, 
#        subsample=0.8, 
#        random_state=123))

HistGX = make_pipeline(
    featurizer, 
    HistGradientBoostingRegressor(
        l2_regularization=0.5, 
        learning_rate=0.05, 
        max_bins=50, 
        max_depth=10, 
        max_iter=200, 
        max_leaf_nodes=20,
        min_samples_leaf=36, 
        random_state=123))

# Define a function to evaluate the models
def evaluate_predictions(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    mae = mean_absolute_error(y_true_exp, y_pred_exp)
    r2 = r2_score(y_true_exp, y_pred_exp)
    return mae, r2

# Train the model
for model_name, model in [
    #                     ('dummy', dummy),  # baseline model
    #                     ('GXboost', GXboost),
                          ('HistGX', HistGX)
                         ]:
    model.fit(X_train, y_train)
    
    # Predict and evaluate for both train and validation sets
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    mae_train, r_2_train = evaluate_predictions(y_train, pred_train)
    mae_validate, r_2_validate = evaluate_predictions(y_test, pred_test)
    
    # print(f"{model_name} - Train Metrics:\n MAE: {mae_train:.4f} R²: {r_2_train:.4f}\n 
    #                        Validate Metrics:\n MAE: {mae_validate:.4f} R²: {r_2_validate:.4f}")


# Tuning parameters for extracting topics in 'abstract' and 'title'

# Parameter grid for abstract topics
# abstract_param_distributions = {
#     "vectorizer__max_features": randint(100, 250),
#     "vectorizer__min_df": [1, 3],  
#     "vectorizer__max_df": [0.7, 0.8], 
#     "vectorizer__ngram_range": [(1, 1), (1, 2)],
#     "lda__n_components": randint(15, 60),  
#     "lda__learning_decay":  [0.6, 0.7],  
#     "lda__doc_topic_prior": [None, 0.1],
#     "lda__topic_word_prior": [None, 0.1],}

# Parameter grid for title topics 
# title_param_distributions = {
#     "vectorizer__max_features": randint(50, 100),
#     "vectorizer__min_df": [1, 3],  
#     "vectorizer__max_df": [0.7, 0.8],  
#     "vectorizer__ngram_range": [(1, 1)],
#     "lda__n_components": randint(15, 40),  
#     "lda__learning_decay": [0.5, 0.7, 0.9], 
#     "lda__doc_topic_prior": [None, 0.1],
#     "lda__topic_word_prior": [None, 0.1],}

# Define pipelines for abstract and title topics
# abstract_pipeline = Pipeline([
#     ("vectorizer", CountVectorizer(
#         analyzer="word",
#         token_pattern=r"\b\w{2,}\b",
#         stop_words="english",
#     )),
#     ("lda", LatentDirichletAllocation(
#         max_iter=20,
#         random_state=123,
#         learning_method="online"))
# ])

# title_pipeline = Pipeline([
#     ("vectorizer", CountVectorizer(
#         analyzer="word",
#         token_pattern=r"\b\w{2,}\b",
#         stop_words="english",
#     )),
#     ("lda", LatentDirichletAllocation(
#         max_iter=20,
#         random_state=123,
#         learning_method="online"
#     ))
# ])

# Initialize random search for abstract and title
# abstract_search = RandomizedSearchCV(
#     estimator=abstract_pipeline,
#     param_distributions=abstract_param_distributions,
#     n_iter=10, 
#     cv=3, 
#     verbose=1,
#     n_jobs=1  
# )

# title_search = RandomizedSearchCV(
#     estimator=title_pipeline,
#     param_distributions=title_param_distributions,
#     n_iter=10,
#     cv=3,
#     verbose=1,
#     n_jobs=1
# )

# Perform the search
# print("Tuning Abstract Topics...")
# abstract_search.fit(X_train['abstract']) 
# print("Tuning Title Topics...")
# title_search.fit(X_train['title']) 

# Print the best parameters  
# print("Best Abstract Topic Params:", abstract_search.best_params_)
# print("Best Title Topic Params:", title_search.best_params_)


# Check the features extracted from the pipeline
# for name, transformer, columns in featurizer.transformers_:
#     print(f"\nTransformer: {name}")
#     if hasattr(transformer, 'get_feature_names_out'):
#         feature_names = transformer.get_feature_names_out()
#         print(f"Feature Names ({name}): {feature_names[:10]}...")   
#         print(f"Number of Features in {name}: {len(feature_names)}")
#     elif hasattr(transformer, 'named_steps'):
     
#         final_step = transformer.named_steps['vectorizer']   
#         if hasattr(final_step, 'get_feature_names_out'):
#             feature_names = final_step.get_feature_names_out()
#             print(f"Feature Names ({name}): {feature_names[:10]}...")
#             print(f"Number of Features in {name}: {len(feature_names)}")
#     else:
#         print(f"Transformer {name} does not provide feature names.")


# Tuning parameters in the pipeline 

# Define parameter grid
# param_distributions = {
    
    # Abstract TF-IDF parameters
#    "featurizer__abstract__ngram_range": [(1, 1), (1, 2)],
#    "featurizer__abstract__max_features": randint(100, 201) ,
#    "featurizer__abstract__min_df": [1, 3],
#    "featurizer__abstract__max_df": [0.7, 0.8],

    # Authors TF-IDF parameters
#    "featurizer__authors__ngram_range": [(1, 1), (1, 2)],
#    "featurizer__authors__max_features": randint(50, 101) ,
#    "featurizer__authors__min_df": [1, 3],
#    "featurizer__authors__max_df": [0.7, 0.8],

    # Title TF-IDF parameters
#    "featurizer__title__ngram_range": [(1, 1), (1, 2)],
#    "featurizer__title__max_features": randint(50, 100),
#    "featurizer__title__min_df": [1, 3],
#    "featurizer__title__max_df": [0.7, 0.8],

     # Venue TF-IDF parameters
#     "featurizer__venue__ngram_range": [(1, 1), (1, 2)],
#     "featurizer__venue__max_features": randint(50, 101),
#     "featurizer__venue__min_df": [1, 3],
#     "featurizer__venue__max_df": [0.7, 0.8], 

     # OneHotEncoder for 'first_author'
#     "featurizer__first_author__max_categories": [10, 15],
#     "featurizer__first_author__handle_unknown": ["ignore", "infrequent_if_exist"],

    # OneHotEncoder for 'venue_transformed'
#     "featurizer__venue_transformed__max_categories": [10, 15],
#     "featurizer__venue_transformed__handle_unknown": ["ignore", "infrequent_if_exist"]

# }

# Initialize random search  
# random_search = RandomizedSearchCV(
#     estimator=HistGX,
#     param_distributions=param_distributions,
#     n_iter=30,   
#     scoring="neg_mean_absolute_error",   
#     cv=3,   
#     verbose=1  
# )

# Fit RandomizedSearchCV
# random_search.fit(X_train, y_train)

# Print the best parameters and the score
# print("Best Parameters:", random_search.best_params_)
# print("Best Score (Neg MAE):", -random_search.best_score_)


# Tuning parameters for the regressor

# Define hyperparameter distributions 
# param_distributions = {
#     "model__learning_rate": [0.01, 0.05],
#     "model__max_iter": [200, 500],
#     "model__max_leaf_nodes": [20, 50],
#     "model__max_depth": [5, 7, 10],
#     "model__min_samples_leaf": randint(10, 50),   
#     "model__l2_regularization": uniform(0.3, 0.7),
#     "model__max_bins": [50, 100, 150]
# }

# Initialize RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=HistGX,
#     param_distributions=param_distributions,
#     n_iter=50, 
#     scoring="neg_mean_absolute_error", 
#     cv=5, 
#     verbose=1, 
#     random_state=123,  
#     n_jobs=1 
# )

# Fit RandomizedSearchCV
# random_search.fit(X_train, y_train)

# Print the best parameters and score
# print("Best parameters found: ", random_search.best_params_)
# print("Best MAE: ", -random_search.best_score_)


# Cross-validation of model performance
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# cv_scores = cross_val_score(
#     HistGX,  
#     X_train,  
#     y_train,  
#     cv=5,  
#     scoring=mae_scorer,  
#     n_jobs=-1  
# )

# print(f"Mean CV MAE: {np.mean(-cv_scores):.4f}")


# Save the fitted pipeline
joblib.dump(HistGX, "HistGX_pipeline_with_lda.pkl")

# Load the pipeline for prediction
loaded_pipeline = joblib.load("HistGX_pipeline_with_lda.pkl")

# Predict the target in the test set
predicted = np.expm1(loaded_pipeline.predict(test_set))
test['n_citation'] = predicted
json.dump(test[['n_citation']].to_dict(orient='records'), open('predicted.json', 'w'), indent=2)

