DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
FILE_NAME = "u.data"
FULL_PATH = paste0(DATA_DIR, FILE_NAME)

source("svd_utils.R")

moviesMelt = cleanData(FULL_PATH)

svdModel = train(moviesMelt)

pred = predict(moviesMelt, svdModel)

r = evaluate(preProcess(moviesMelt), preProcess(as.data.frame(pred)))
# RMSE IS 1.344 with old preprocessing
# RMSE is .821 with new preprocessing

