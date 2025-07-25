# Predicting Scientific Paper Citations

## Overview

This project was completed for the individual assignment of Machine Learning course.  
The task was to develop a predictive model that estimates the number of citations a scientific paper would receive based on its metadata.

The assignment was structured as a competitive challenge, where students submitted predictions and were ranked on a leaderboard based on their model's Mean Absolute Error (MAE).



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

## Methods

Predictions were evaluated using **Mean Absolute Error (MAE)** 

Title

Description of technology
Why?

Description of the process
Why?

## Evaluation 

The model needed to be applied on the test data to generate predictions and submit the predictions to a central leaderboard. 

This submission received bonus points for having scored higher than the baseline on the leaderboard. 



Table of contents

extras
How the project came about
The motivation
Limitations
Challenges
What problem it hopes to solve
What the intended use is
Credits
