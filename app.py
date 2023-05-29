from flask import Flask, request
import pandas as pd
import joblib
import numpy as np
app = Flask(__name__)
def robust_data(path):
    import json
    # robust & dummy data set test
    with open(path, 'r') as f:
        json = json.load(f)
    columns = ['powerPS', 'odometer', 'Age']
    # chứa các giá trị trung bình, std theo columns trên
    IQRs_array = [json['robust']['IQRs'][col] for col in columns]
    median_array = [json['robust']['median'][col] for col in columns]
    return IQRs_array, median_array
@app.route('/api/predict/csv')
def predict_csv():
    try:
        if 'file' not in request.files:
            return 'No file uploaded', 400
        # use ordinal dummy encoding
        file = request.files['file']
        predict = pd.read_csv(file.stream, encoding='utf-8', index_col=False)
        prefix = pd.read_csv('./prefix.csv', encoding='utf-8', index_col=False)
        df = pd.concat([predict, prefix], ignore_index=True)
        df = pd.get_dummies(df, drop_first=True, dtype=int)
        print(df.columns)
        drop = len(prefix)
        df = df.iloc[:-drop]

        IQRs, median = robust_data('./settings/data.json')
        columns = ['powerPS', 'odometer', 'Age']
        #(X - median(X)) / IQR(X)
        df[columns] = (df[columns] - median)/IQRs

        # Load the model from the file
        GDBT_Robust_Scaled = joblib.load('./models/Gradient_Boosting_scaled.pkl')
        predicted_list = GDBT_Robust_Scaled.predict(df)
        predict['predicted'] = np.exp(predicted_list)
        return predict.to_csv(index=False), 200

    except Exception as e:
        return "internal err", 500

if __name__ == '__main__':
    app.run()