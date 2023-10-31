from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle

seasons = np.zeros(4)
months  = np.zeros(12)
hours   = np.zeros(24)
holidays = np.zeros(2)
days = np.zeros(7)

app = Flask(__name__,static_folder='static')

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/predict",methods=['GET','POST'])
def predict():

    season  =  int(request.args.get("season"))
    month   =  int(request.args.get("month"))
    holiday = int(request.args.get("holiday"))
    hour    = int(request.args.get("hour"))
    day     = int(request.args.get("weekday"))
    temperature = float(request.args.get("temperature"))
    humidity = int(request.args.get("humidity"))
    windSpeed = float(request.args.get("windSpeed"))
    visibility = float(request.args.get("visibility"))
    dewPoint = float(request.args.get("dewPoint"))
    solarRadiation = float(request.args.get("solarRadiation"))
    rainfall = float(request.args.get("rainfall"))
    snowfall = float(request.args.get("snowfall"))


    seasons[season]=1
    months[month]=1
    hours[hour]=1
    if(holiday==0):
      holidays[0]=1 
    else:
      holidays[1]=1
    days[day]=1

    temp = np.array([temperature])
    humid = np.array([humidity])
    wspeed = np.array([windSpeed])
    visibile = np.array([visibility])
    dew = np.array([dewPoint])
    solar = np.array([solarRadiation])
    rain = np.array([rainfall])
    snow = np.array([snowfall])

    features = np.concatenate((seasons,
                    holidays,
                    months,days,hours,
                    temp,humid,wspeed,
                    visibile,dew,solar,rain,snow)).reshape(1,-1)
    


    model = pickle.load(open("model.pkl","rb"))
    print(features)
    prediction = int(model.predict(features))
    return render_template("index.html",prediction_text="Predicted number of rented bikes is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)