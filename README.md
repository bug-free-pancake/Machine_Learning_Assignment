# ML Challenge: Predicting Scientific Paper Citations

## Overview

This project was completed for the Machine Learning individual assignment.  
The task was to develop a predictive model that estimates the number of citations a scientific paper would receive based on its metadata.

The assignment was structured as a competitive challenge hosted on Codalab, where students submitted predictions and were ranked on aleaderboard based on their model's Mean Absolute Error (MAE).
The submission received bonus points for having scored higher than the baseline on the leaderboard. 

## Task Description

- **Input**: Metadata of scientific papers (from `train.json` and `test.json`)
- **Output**: Predicted number of citations for each paper in the test set (`predicted.json`)
- **Goal**: Minimize the Mean Absolute Error (MAE) between the predicted and actual citations

## Dataset

The dataset included:
- `train.json`: List of paper records including the number of citations under the key `n_citation`
- `test.json`: Similar to training data, but without `n_citation`; the goal was to predict this value

Each record is a dictionary with fields such as:
- Title
- Authors
- Abstract
- Year of publication
- Venue
- etc.

## Evaluation

Predictions were evaluated using **Mean Absolute Error (MAE)** 
