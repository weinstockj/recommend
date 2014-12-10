library("reshape2")
library("SpatioTemporal")
# READ IN DATA
cleanData = function(file){
  movies = read.table(file, header = F)
  names(movies) = c("user", "item", "rating", "timestamp")
  
  # EXCLUDE TIMESTAMP
  movies = movies[-4]
  
  # get in matrix/adjacency format
  moviesMelt = dcast(data = movies, user ~ item, value.var = "rating")
#   
#   # impute mean of each column
#   moviesMelt[1:length(moviesMelt)] = sapply(moviesMelt, function(x) ifelse(is.na(x), 
#     mean(x, na.rm = T), x))
  return(list(moviesMelt[, 1], moviesMelt[, -1]))
}

impute = function(df){
  df = apply(df, 1:2, function(x) ifelse(is.na(x), 0, x))
}

preProcess = function(df, retScales = T){
  # CONVERT EVERY COLUMN TO Z-SCORES
  OVERALL_MEAN = mean(as.matrix(df), na.rm = T)
  scaled = apply(df[, 1:ncol(df)], 1:2, function(x) x - OVERALL_MEAN)
  scaledcenters = colMeans(scaled, na.rm = T)
  scaled = scaled - rep(scaledcenters, rep.int(nrow(scaled), ncol(scaled)))
  scaledRows = rowMeans(scaled, na.rm = T)
  scaled = t(t(scaled) - rep(scaledRows, rep.int(ncol(scaled), nrow(scaled))))
  if(retScales) return(list(scaled, OVERALL_MEAN, scaledcenters, scaledRows))
  else return(scaled)
}

deNormalize = function(df, overall, cols, rows){
  scaled = apply(df[, 1:ncol(df)], 1:2, function(x) x + OVERALL_MEAN)
  scaledcenters = colMeans(scaled, na.rm = T)
  scaled = scaled + rep(cols, rep.int(nrow(scaled), ncol(scaled)))
  scaled = t(t(scaled) + rep(rows, rep.int(ncol(scaled), nrow(scaled))))
  return(scaled)
}

train = function(df){
  #   SVDmiss(df, niter = 25)$svd
  svd(df)
}

predict = function(newdata, model, number = ceiling(nrow(newdata)/ 400)){
  U = model$u
  V = model$v
  d = model$d
  N_ROWS = nrow(newdata)
  N_COLS = ncol(newdata)
  idx = expand.grid(1:N_ROWS, 1:N_COLS)
  FUN = function(x){
    user = x[[1]]
    item = x[[2]]
    userVec = U[, user][1:number] 
    itemVec = V[item, ][1:number] 
    return(userVec %*% itemVec) # dot product of item and user vectors
  }
  res = apply(idx, 1, FUN)
  res = matrix(res, nrow = N_ROWS, ncol = N_COLS)
  return(res)
}

# evaluate = function(data, predictions){
#   N_ROWS_DATA = nrow(data)
#   N_COLS_DATA = ncol(data)
#   N_ROWS_PRED = nrow(predictions)
#   N_COLS_PRED = ncol(predictions) 
#   stopifnot(N_ROWS_DATA == N_ROWS_PRED & N_COLS_DATA == N_COLS_PRED)
#   N_ELEMENTS = N_ROWS_DATA * N_COLS_DATA
#   res = rep(NA, N_ELEMENTS)
#   index = 1
#   # CALCULATE SQUARED DEVIATIONS
#   for(i in 1:N_ROWS_DATA) {
#     for(j in 1:N_COLS_DATA) {
#       actual = data[i, j]
#       pred = predictions[i, j]
#       if(abs(pred) > 20){
#         pred = NA
#       }
#       res[index] = (actual - pred) ^ 2
#       index = index + 1
#     }
#   }
#   cat(sprintf("\nRMSE is %.3f", sqrt(mean(res, na.rm = T))))
#   invisible(res)
# }

evaluate = function(data, predictions){
  N_ROWS_DATA = nrow(data)
  N_COLS_DATA = ncol(data)
  N_ROWS_PRED = nrow(predictions)
  N_COLS_PRED = ncol(predictions)
  stopifnot(N_ROWS_DATA == N_ROWS_PRED & N_COLS_DATA == N_COLS_PRED)
  datavec = as.vector(data)
  predvec = as.vector(predictions)
  # CALCULATE SQUARED DEVIATIONS
  res = (datavec - predvec) ^ 2
  cat(sprintf("\nRMSE is %.3f", sqrt(mean(res, na.rm = T))))
  invisible(res)
}

resize = function(data, predictions){
  N_ROWS_DATA = nrow(data)
  N_COLS_DATA = ncol(data)
  N_ROWS_PRED = nrow(predictions)
  N_COLS_PRED = ncol(predictions)
  if(N_ROWS_DATA > N_ROWS_PRED){
    DIFF_ROWS = N_ROWS_DATA - N_ROWS_PRED
    addMat = matrix(NA, nrow = DIFF_ROWS, ncol = N_COLS_PRED)
    colnames(addMat) = colnames(predictions)
    predictions = rbind(predictions, addMat)
    N_ROWS_PRED = nrow(predictions)
  }
  if(N_COLS_DATA > N_COLS_PRED){
    DIFF_COLS = N_COLS_DATA - N_COLS_PRED
    addMat = matrix(NA, nrow = N_ROWS_PRED, ncol = DIFF_COLS)
    colnames(addMat) = (ncol(predictions) + 1):N_COLS_DATA
    predictions = cbind(predictions, addMat)
  }
  return(predictions)
}
