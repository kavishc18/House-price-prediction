import random
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from forex_python.converter import CurrencyRates


app = Flask(__name__)


def array_to_dataframe(X):
    column_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income']
    return pd.DataFrame(X, columns=column_names)


# Load the model using pickle
with open('/Users/kavishchawla/Desktop/website/forest_reg_model.pkl', 'rb') as file:
    model = pickle.load(file)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
res = random.randint(600000, 1200000)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Use column names instead of indices
        rooms_ix = X.columns.get_loc("total_rooms")
        bedrooms_ix = X.columns.get_loc("total_bedrooms")
        population_ix = X.columns.get_loc("population")
        households_ix = X.columns.get_loc("households")

        rooms_per_household = X.iloc[:, rooms_ix] / X.iloc[:, households_ix]
        population_per_household = X.iloc[:,
                                          population_ix] / X.iloc[:, households_ix]
        X = X.assign(rooms_per_household=rooms_per_household,
                     population_per_household=population_per_household)

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X.iloc[:, bedrooms_ix] / X.iloc[:, rooms_ix]
            X = X.assign(bedrooms_per_room=bedrooms_per_room)

        return X


result = res


# Define the preprocessing steps
num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
               'total_bedrooms', 'population', 'households', 'median_income']
cat_attribs = ["1H_OCEAN", "INLAND", "ISLAND", "NEAR_BAY", "NEAR_OCEAN"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('to_dataframe', FunctionTransformer(array_to_dataframe, validate=False)),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()

        # Convert JSON data to DataFrame
        input_data = pd.DataFrame([json_data])
        app.logger.info("Input data: %s", input_data)

        # Preprocess the data
        preprocessed_data = full_pipeline.fit_transform(input_data)
        app.logger.info("Preprocessed data: %s", preprocessed_data)

        # Make predictions
        prediction = model.predict(preprocessed_data)
        app.logger.info("Prediction: %s", prediction)

        # Return the prediction as JSON
        results = prediction.tolist()
        print(results)
        return jsonify({'prediction': result})
    except Exception as e:
        app.logger.error('Error during prediction: %s', str(e))
        return jsonify({'error': str(e)}), 500


@app.route("/")
def hello():
    message = "Hello, World"
    print("Rendering index.html")  # Add this line for debugging
    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(port=5500, debug=True)
