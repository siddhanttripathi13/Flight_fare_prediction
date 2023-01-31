# Import necessary libraries
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the regression model and encoders
model = pickle.load(open('model.pkl', 'rb'))
airline_encoder = pickle.load(open('airline_encoder.pkl', 'rb'))
source_encoder = pickle.load(open('source_encoder.pkl', 'rb'))
destination_encoder = pickle.load(open('destination_encoder.pkl', 'rb'))


@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():

    if request.method == 'POST':
        
        # Date_of_Journey
        dep_date = request.form['Dep_Time']
        dep_date = pd.to_datetime(dep_date)
        journey_day = int(dep_date.day)
        journey_month = int(dep_date.month)
        journey_dayofweek = int(dep_date.dayofweek)

        # Departure
        dep_hour = int(dep_date.hour)
        dep_min = int(dep_date.minute)

        # Arrival
        arr_date = request.form['Arrival_Time']
        arr_date = pd.to_datetime(arr_date)
        arr_hour = int(arr_date.hour)
        arr_minute = int(arr_date.minute)

        # Duration
        duration = arr_date - dep_date
        dur_hours = duration.total_seconds()//3600
        dur_min = (duration.total_seconds() - (dur_hours*3600))//60

        # Total stops
        total_stops = int(request.form['stops'])

        # Initial input
        data = [total_stops, journey_day, journey_month,
                journey_dayofweek, dep_hour, dep_min,
                arr_hour, arr_minute, dur_hours, dur_min]

        # Airline
        airline = request.form['airline']
        data += list(airline_encoder.transform(
            [[airline]]).toarray().flatten())

        # Source
        source = request.form['Source']
        data += list(source_encoder.transform([[source]]).toarray().flatten())

        # Destination
        destination = request.form['Destination']
        data += list(destination_encoder.transform(
            [[destination]]).toarray().flatten())

        # Final input and prediction
        data = np.array([data]).reshape(1, -1)
        prediction = round(model.predict(data)[0], 2)

        return render_template('home.html', prediction_text=f'Your flight price is Rs. {prediction}')

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
