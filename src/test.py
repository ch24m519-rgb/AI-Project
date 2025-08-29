import requests
import json

url = "http://localhost:5050/predict"

data = {
    "PassengerId": 893,
    "Pclass": 3,
    "Name": "Wilkes, Mrs. James (Ellen Needs)",
    "Sex": "female",
    "Age": 47.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "363272",
    "Fare": 7.0,
    "Cabin": None,
    "Embarked": "S"
}

json_data = json.dumps(data)

response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

if response.status_code == 200:
    print("Prediction: ", response.json()['prediction'])
else:
    error_msg = response.json().get('error', response.text)
    print("Error: ", error_msg)
