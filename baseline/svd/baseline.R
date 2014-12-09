DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
FILE_NAME = "u.data"
# FILE_NAME = "u1.base"
FULL_PATH = paste0(DATA_DIR, FILE_NAME)

source("svd_utils.R")

moviesMelt = cleanData(FULL_PATH)
USER_ID = moviesMelt[[1]]
moviesMelt = moviesMelt[[2]]
moviesMeltNorm = preProcess(moviesMelt)
OVERALL_MEAN = moviesMeltNorm[[2]]
COL_MEANS = moviesMeltNorm[[3]]
ROW_MEANS = moviesMeltNorm[[4]]
moviesMeltNorm = moviesMeltNorm[[1]]
moviesMeltNorm = impute(moviesMeltNorm)

svdModel = train(moviesMeltNorm)

pred = predict(moviesMelt, svdModel, 5)
pred = preProcess(as.data.frame(pred), F)
pred = deNormalize(pred, OVERALL_MEAN, COL_MEANS, ROW_MEANS)

r = evaluate(moviesMelt, pred)
