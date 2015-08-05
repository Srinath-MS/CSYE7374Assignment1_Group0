__author__ = 'shwetaanchan'

from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint , LinearRegressionWithSGD, RidgeRegressionWithSGD, LassoWithSGD


sc = SparkContext()

lines = sc.textFile("/Users/shwetaanchan/PycharmProjects/Learning/WineQuality.csv")
print (lines.take(5))
header = lines.take(1)[0]
rows = lines.filter(lambda line: line != header)

def append_binary(line):
    fields = [float(x) for x in line.split(',')]
    if (int(fields[11])>=6):
        fields[11]=1
    else:
        fields[11]=0
    return LabeledPoint(fields[11],fields[0:10])

data = rows.map(lambda line: append_binary(line))
print (data.take(10))


training, test = data.randomSplit([0.6, 0.4], seed = 0)

def evaluate_model(test,model):
    labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
    testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
    return (testErr)

### SVM with SVG
model_svm_l2 = evaluate_model(test,SVMWithSGD.train(training,iterations=100,regType="l2"))
model_svm_l1 = evaluate_model(test,SVMWithSGD.train(training,iterations=100,regType="l1"))
model_svm_l0 = evaluate_model(test,SVMWithSGD.train(training,iterations=100,regType=None))

## Logistic Regression with LBFGS
model_log_lbfgs_l2 = evaluate_model(test,LogisticRegressionWithLBFGS.train(training,iterations=100,regType="l2"))
model_log_lbfgs_l1 = evaluate_model(test,LogisticRegressionWithLBFGS.train(training,iterations=100,regType="l1"))
model_log_lbfgs_l0 = evaluate_model(test,LogisticRegressionWithLBFGS.train(training,iterations=100,regType=None))

## Logistic Regression with SGD
model_log_sgd_l2 = evaluate_model(test,LogisticRegressionWithSGD.train(training,iterations=100,regType="l2"))
model_log_sgd_l1 = evaluate_model(test,LogisticRegressionWithSGD.train(training,iterations=100,regType="l1"))
model_log_sgd_l0 = evaluate_model(test,LogisticRegressionWithSGD.train(training,iterations=100,regType=None))


############################### Regression #######################

def parse_line(line):
    fields = [float(x) for x in line.split(‘,’)]
    return LabeledPoint(fields[11],fields[0:10])

data_reg = rows.map(lambda line: parse_line(line))

# Split data aproximately into training (60%) and test (40%)
training_reg, test_reg = data_reg.randomSplit([0.6, 0.4], seed = 0)

print (training_reg.take(5))

def evaluate_model_reg(test,model):
    valuesAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    return (MSE)


### LinearRegression with SGD
model_lreg_sgd_l2 = evaluate_model_reg(test_reg,LinearRegressionWithSGD.train(training_reg,iterations=1000,step=0.0001,regType="l2"))
model_lreg_sgd_l1 = evaluate_model_reg(test_reg,LinearRegressionWithSGD.train(training_reg,iterations=1000,step=0.0001,regType="l1"))
model_lreg_sgd_l0 = evaluate_model_reg(test_reg,LinearRegressionWithSGD.train(training_reg,iterations=1000,step=0.0001,regType=None))

### RidgeRegression
model_ridge = evaluate_model_reg(test_reg,RidgeRegressionWithSGD.train(training_reg,iterations=1000, step=0.0001))

### Lasso
model_lasso = evaluate_model_reg(test_reg,LassoWithSGD.train(training_reg,iterations=1000, step =0.0001))

#################### OUTPUTS #################################
print("Testing Error :"+"model_svm_l2 = " + str(model_svm_l2))
print("Testing Error :"+"model_svm_l1 = " + str(model_svm_l1))
print("Testing Error :"+"model_svm_l0 = " + str(model_svm_l0))

print("Testing Error :"+"model_log_lbfgs_l2 = " + str(model_log_lbfgs_l2))
print("Testing Error :"+"model_log_lbfgs_l1 = " + str(model_log_lbfgs_l1))
print("Testing Error :"+"model_log_lbfgs_l0 = " + str(model_log_lbfgs_l0))

print("Testing Error :"+"model_log_sgd_l2 = " + str(model_log_sgd_l2))
print("Testing Error :"+"model_log_sgd_l1 = " + str(model_log_sgd_l1))
print("Testing Error :"+"model_log_sgd_l0 = " + str(model_log_sgd_l0))

print("MSE Error :"+"model_lreg_sgd_l2 = " + str(model_lreg_sgd_l2))
print("MSE Error :"+"model_lreg_sgd_l1 = " + str(model_lreg_sgd_l1))
print("MSE Error :"+"model_lreg_sgd_l0 = " + str(model_lreg_sgd_l0))

print("MSE Error :"+"model_ridge = " + str(model_ridge))
print("MSE Error :"+"model_lasso = " + str(model_lasso))












