DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
N_FOLDS = 5
source("svd_utils.R")

rmseVector = rep(0, N_FOLDS)
library("plyr")
source("svd_utils.R")
DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
N_FOLDS = 5
N_VEC = c(1, 2, 5, 10, 50, 100, 200)

rmseVector = list()
index = 1
for(nvec in N_VEC){
  for(i in 1:N_FOLDS){
    cat("\nfold is: ", i)
    trainFile = sprintf("u%d.base", i)
    testFile = sprintf("u%d.test", i)
    trainPath = paste0(DATA_DIR, trainFile)
    testPath = paste0(DATA_DIR, testFile)
    trainData = cleanData(trainPath)[[2]]
    testData = cleanData(testPath)[[2]]
    testData = resize(trainData, testData)
    trainDataNorm = preProcess(trainData)
    OVERALL_MEAN = trainDataNorm[[2]]
    COL_MEANS = trainDataNorm[[3]]
    ROW_MEANS = trainDataNorm[[4]]
    trainDataNorm = trainDataNorm[[1]]
    trainDataNorm = impute(trainDataNorm)  
    svdModel = train(trainDataNorm)
    pred = predict(testData, svdModel, nvec)
    pred = preProcess(as.data.frame(pred), F)
    pred = deNormalize(pred, OVERALL_MEAN, COL_MEANS, ROW_MEANS)
    res = evaluate(testData, pred)
    rmseVector[[length(rmseVector) + 1]] = data.frame(rmse = sqrt(mean(res, na.rm = T)), 
      nvec = nvec)
    index = index + 1
  }
}
