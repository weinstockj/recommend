DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
N_FOLDS = 5
source("svd_utils.R")

rmseVector = rep(0, N_FOLDS)
for(i in 1:N_FOLDS){
  cat("\nfold is: ", i)
  trainFile = sprintf("u%d.base", i)
  testFile = sprintf("u%d.test", i)
  trainPath = paste0(DATA_DIR, trainFile)
  testPath = paste0(DATA_DIR, testFile)
  trainData = cleanData(trainPath)
  testData = cleanData(testPath)
  svdModel = train(trainData)
  pred = predict(testData, svdModel, 5)
  res = evaluate(preProcess(testData), preProcess(as.data.frame(pred)))
  rmseVector[i] = sqrt(mean(res, na.rm = T))
}

# old preprocessing
# vectors = 30 -> 1.332
# vectors = 10 -> 1.332
# vectors = 3 -> 1.332184
# vectors = 1 -> 1.332117
# new preprocessing
# all vectors -> 0.8563558
# vectors = 30 -> 0.8560269
# vectors = 5 -> 0.8560171