import joblib

input = [5.8,2.8,5.1,2.4]
model_load = joblib.load('model.pkl')
prediction = model_load.predict([input])

print("Prediction: ", prediction)