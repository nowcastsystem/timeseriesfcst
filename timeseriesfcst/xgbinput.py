import xgboost as xgb



class XGBInput(object):
    def __init__(self, label, covariate, splitdt):
        self.label = label
        self.covariate = covariate

        self.labeltrain = self.label.loc[self.label.index <= splitdt].copy()
        self.covariatetrain = self.covariate.loc[self.covariate.index <= splitdt].copy()

        self.labeltest = self.label.loc[self.label.index > splitdt].copy()
        self.covariatetest = self.covariate.loc[self.covariate.index > splitdt].copy()

        self.dtrain = xgb.DMatrix(self.covariatetrain, label=self.labeltrain)
        self.trainindex = self.covariatetrain.index
        self.dtest = xgb.DMatrix(self.covariatetest, label=self.labeltest)
        self.testindex = self.covariatetest.index
        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]

        self.dtraintest = xgb.DMatrix(covariate, label=label)
        self.traintestindex = covariate.index


    def gettrain(self):
        return self.dtrain

    def gettest(self):
        return self.dtest