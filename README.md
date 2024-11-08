# Predictive Yelp Recommender: Suggesting Relevant Businesses Based on User Behavior

## Overview
This PySpark script is designed to train a predictive model using the XGBoost algorithm for a Yelp dataset. The trained model is then used to make predictions on a validation dataset, and the results are saved to an output file.

## Prerequisites
- Apache Spark: Make sure you have Apache Spark installed, and the `pyspark` library is available in your Python environment.

## Usage
1. **Install Required Packages:** Ensure that the necessary Python packages are installed by running:
`pip install pyspark numpy xgboost scikit-learn joblib`

2. **Run the Script:** Execute the script using the following command:
`spark-submit your_script_name.py <folder_path> <validation_file_path> <output_file_path>`
- `<folder_path>`: The path to the folder containing the Yelp dataset files (`yelp_train.csv`, `user.json`, `business.json`, `tip.json`, `photo.json`).
- `<validation_file_path>`: The path to the validation dataset file.
- `<output_file_path>`: The path to save the predictions.

3. **Output:** The script generates a CSV file with predicted ratings for user-business pairs.

## Script Details
- **Data Loading:** The script loads Yelp dataset files (`yelp_train.csv`, `user.json`, `business.json`, `tip.json`, `photo.json`) using Spark RDDs.
- **Feature Extraction:** Features are extracted for users, businesses, tips, and photos.
- **Model Training:** The XGBoost model is trained using the training dataset (`yelp_train.csv`).
- **Model Prediction:** The trained model is used to predict ratings on the validation dataset.
- **Output:** Predicted ratings for user-business pairs are saved in a CSV file.

## Model Tuning (Optional)
- The script includes a commented section for hyperparameter tuning using RandomizedSearchCV. Uncomment and adjust the parameters if you want to perform tuning.
- The best model is saved with the specified hyperparameters.

## Files and Dependencies
- The script relies on the Yelp dataset files (`yelp_train.csv`, `user.json`, `business.json`, `tip.json`, `photo.json`).
- Python libraries used: `pyspark`, `numpy`, `xgboost`, `scikit-learn`, `joblib`.

## Acknowledgments
- This script utilizes Apache Spark for distributed computing and XGBoost for machine learning.

## Author
Revati Pawar

## License
This project is licensed under the [MIT License](LICENSE).
