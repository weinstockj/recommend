DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
N_FOLDS = 5
N_HIDDEN = c(2, 5, 10, 30, 50)
source("svd_utils.R")

rmseVectorRbm = list()
for(nh in N_HIDDEN){
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
    #   rbmModel = rbm.train(trainData[, -1][, 1:ncol(testData)], hidden = N_HIDDEN)
    #   h = rbm.up(rbmModel, as.matrix(testData[, -1])[, 1:(ncol(testData) - 1)])
    rbmModel = rbm.train(trainDataNorm, hidden = nh)
    h = rbm.up(rbmModel, as.matrix(impute(testData)))
    pred = rbm.down(rbmModel, h)
    pred = preProcess(as.data.frame(pred), F)
    pred = deNormalize(pred, OVERALL_MEAN, COL_MEANS, ROW_MEANS)
    res = evaluate(testData, pred)
    rmseVectorRbm[[length(rmseVectorRbm) + 1]] = data.frame(rmse = sqrt(mean(res, na.rm = T)), 
      hidden = nh)
    index = index + 1
  }
}


