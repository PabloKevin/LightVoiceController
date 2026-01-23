from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import r2_score

model_path = "model_weights/audioProcessingModel.keras"
model = load_model(model_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../..", "DataSets/RawAudio2MFCC")
X_test = np.load(os.path.join(data_path, "X_train.npy"))
y_test = np.load(os.path.join(data_path, "Y_train.npy"))

#mean_yTrain = 9.795799596756644e-10
#std_yTrain = 0.0014433130028423165
#Y_predicted = Y_predicted * std_yTrain + mean_yTrain

Xmean=0.8719726204872131 
Xstd=0.179367333650589
X_test = X_test * Xstd + Xmean

Y_predicted = model.predict(X_test)

Ymean=9.795799596756644e-10 
Ystd=0.0014433130028423165
Y_predicted = Y_predicted * Ystd + Ymean

Ymean=9.795799596756644e-10 
Ystd=0.0014433130028423165
y_test = y_test * Ystd + Ymean

r2 = r2_score(y_test, Y_predicted)
print(f"Coeficiente de determinación R²: {r2:.4f}")

save_path = os.path.join(BASE_DIR, "../..", "DataSets/RawAudio2MFCC/Y_predicted.npy")
#mean_yTrain = -47.319750827110255
#std_yTrain = 174.86826506622137
#Y_predicted = Y_predicted * std_yTrain + mean_yTrain
#np.save(save_path, Y_predicted)