library("reshape2")

DATA_DIR = "~/2nd semester 2014/machine learning/project/implementation/data/ml-100k/ml-100k/"
FILE_NAME = "u.data"

# READ IN DATA
movies = read.table(paste0(DATA_DIR, FILE_NAME), header = F)
names(movies) = c("user", "item", "rating", "timestamp")

# EXCLUDE TIMESTAMP
movies = movies[-4]

# get in matrix/adjacency format
moviesMelt = dcast(data = movies, user ~ item, value.var = "rating")

# impute mean of each column
moviesMelt[1:length(moviesMelt)] = sapply(moviesMelt, function(x) ifelse(is.na(x), 
  mean(x, na.rm = T), x))

preProcess = function(df){
  # CONVERT EVERY COLUMN TO Z-SCORES
  USER_ID = df[, 1]
  scaled = sapply(df[, 2:ncol(df)], function(x) {
    SD = sd(x, na.rm = T)
    if(SD == 0){
      return(x - mean(x, na.rm = T))
    } else {
      return((x - mean(x, na.rm = T)) / sd(x, na.rm = T))
    }
  })
  return(cbind(USER_ID, scaled))
}

train = function(df, scaleData = T){
  if(scaleData){
    df = preProcess(df)
  }
  userIndex = which(names(moviesMelt) == "user")
  svd(df[, -userIndex])
}

svdResults = train(moviesMelt)

predict = function(newdata, model){
  U = model$u
  V = model$v
  N_ROWS = nrow(newdata)
  N_COLS = ncol(newdata) - 1 # because of id column
  idx = expand.grid(1:N_ROWS, 1:N_COLS)
  FUN = function(x){
    user = x[[1]]
    item = x[[2]]
    userVec = U[, user]
    itemVec = V[item, ]
    return(userVec %*% itemVec) # dot product of item and user vectors
  }
  res = apply(idx, 1, FUN)
  res = matrix(res, nrow = N_ROWS, ncol = N_COLS)
  res = cbind(newdata[, 1], res)
}

pred = predict(moviesMelt, svdResults)

evaluate = function(data, predictions){
  N_ROWS_DATA = nrow(data)
  N_COLS_DATA = ncol(data) - 1
  N_ROWS_PRED = nrow(predictions)
  N_COLS_PRED = ncol(predictions) - 1
  stopifnot(N_ROWS_DATA == N_ROWS_PRED & N_COLS_DATA == N_COLS_PRED)
  N_ELEMENTS = N_ROWS_DATA * N_COLS_DATA
  res = rep(NA, N_ELEMENTS)
  index = 1
  # CALCULATE SQUARED DEVIATIONS
  for(i in 1:N_ROWS_DATA) {
    for(j in 2:N_COLS_DATA) {
      actual = data[i, j]
      pred = predictions[i, j]
      res[index] = (actual - pred) ^ 2
      index = index + 1
    }
  }
  cat(sprintf("RMSE is %.3f\n", sqrt(mean(res, na.rm = T))))
  invisible(res)
}

r = evaluate(preProcess(moviesMelt), preProcess(as.data.frame(pred)))
# RMSE IS 1.344

