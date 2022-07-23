import os                   # For interacting with the file system
import pandas as pd         # For data wrangling operations
import pickle               # For loading the pickled model pipeline
from flask import Flask, render_template,  request, url_for             # For web application development

app = Flask(__name__)

# Home route
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')        #render_template helps generate output of our HTML file

# Prediction route
@app.route("/predict",  methods=['POST'])
def predict():
    # Extract the form values:
    id = request.form['form-id']
    warehouse_block = request.form["form-warehouse-block"]
    mode_of_shipment = request.form["form-shipment-mode"]
    customer_care_calls = request.form["form-customer-care-calls"]
    customer_ratings = request.form["form-customer-ratings"]
    cost_of_product = request.form["form_product_cost"]
    prior_purchases = request.form["form-prior-purchases"]
    product_importance = request.form["form_product_importance"]
    gender = request.form["form-gender"]
    discount_offered = request.form["form-discount-offered"]
    weights_gms = request.form["form-weights-gms"]
    form_data = {
        "ID": [id],
        "Warehouse_block": [warehouse_block],
        "Mode_of_Shipment": [mode_of_shipment],
        "Customer_care_calls": [customer_care_calls],
        "Customer_rating": [customer_ratings],
        "Cost_of_the_Product": [cost_of_product],
        "Prior_purchases": [prior_purchases],
        "Product_importance": [product_importance],
        "Gender": [gender],
        "Discount_offered": [discount_offered],
        "Weight_in_gms": [weights_gms]
    }
    # create form dataframe to apply model pipeline
    form_df = pd.DataFrame(form_data)
    form_df.drop(columns=['ID'], inplace=True)

    # loading model pipeline
    model_pipeline = pickle.load(open("pipeline.pkl", "rb"))
    # applying model pipeline from form dataframe to make predictions
    my_prediction = model_pipeline.predict(form_df)
    return render_template('index.html', prediction=my_prediction)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host ='0.0.0.0', port = port)