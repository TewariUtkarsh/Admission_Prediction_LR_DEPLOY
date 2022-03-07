import pickle
import json

class LinearRegressionOperation:

    def __init__(self, x_user):
        self.lr = pickle.load(open("lr_model.pickle", 'rb'))
        self.x_user = x_user
        self.par_x_user = None
        self.LR_score = None

    def getScores(self):
        scores = [self.getPrediction(), self.getRsquareLR(), self.getAdjRsqaureLR(), self.getRsquareL1(),
                  self.getRsquareL2(), self.getRsquareElasticNet()]
        return scores


    def parseInput(self):
        temp = self.x_user.split(",")
        self.par_x_user = [eval(i) for i in temp]

    def getPrediction(self):
        scaler = pickle.load(open("scaler_obj.py", 'rb'))
        std_x_user = scaler.transform([self.par_x_user])
        y_pred = self.lr.predict(std_x_user)
        return y_pred[0][0]


    def getRsquareLR(self):
        s = json.load(open("score.json", 'r'))
        return s["R-Square Score(LR)"]

    def getAdjRsqaureLR(self):
        s = json.load(open("score.json", 'r'))
        return s["Adjusted R-Square Score(LR)"]

    def getRsquareL1(self):
        s = json.load(open("score.json", 'r'))
        return s["R-Square Score(L1)"]

    def getRsquareL2(self):
        s = json.load(open("score.json", 'r'))
        return s["R-Square Score(L2)"]

    def getRsquareElasticNet(self):
        s = json.load(open("score.json", 'r'))
        return s["R-Square Score(Elastic Net)"]