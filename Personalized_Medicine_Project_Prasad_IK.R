#Remove all objects from R workspace
rm(list=ls())

#Set current working directory where the data_set files are stored
#setwd("C:/Users/Desktop/Project2_Prasad_IK/Data_Set")

#Load packages required to perform certain functions
library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(stringr)
library(tm)
library(syuzhet) 

#Load the data into R
train_text <- do.call(rbind,strsplit(readLines('training_text'),'||',fixed=T))
train_text <- as.data.table(train_text)
train_text <- train_text[-1,]
colnames(train_text) <- c("ID", "Text")
train_text$ID <- as.numeric(train_text$ID)

test_text <- do.call(rbind,strsplit(readLines('test_text'),'||',fixed=T))
test_text <- as.data.table(test_text)
test_text <- test_text[-1,]
colnames(test_text) <- c("ID", "Text")
test_text$ID <- as.numeric(test_text$ID)

train <- fread("training_variants", sep=",", stringsAsFactors = T)
test <- fread("test_variants", sep=",", stringsAsFactors = T)
train <- merge(train,train_text,by="ID")
test <- merge(test,test_text,by="ID")
rm(test_text,train_text)
gc()      #garbage collection

test$Class <- -1
data_full <- rbind(train,test)   #Combine the tables train and test
rm(train,test)
gc()      #garbage collection


#Due to space and time issue I am considering part of the data
#In case of good space we can consider data_full
data1=data_full[1:1000,]
data2=data_full[3322:3600]
data=rbind(data1,data2)


#Extracting additional features number of characters and words from text data
data$no_char <- as.numeric(nchar(data$Text))
data$no_words <- as.numeric(str_count(data$Text, "\\S+"))

#Pre-Processing of text data
text <- Corpus(VectorSource(data$Text))
text <- tm_map(text, stripWhitespace)
text <- tm_map(text, content_transformer(tolower))
text <- tm_map(text, removePunctuation)
text <- tm_map(text, removeWords, stopwords("english"))
oth_stopwords=c("also","name","much","new","next","none","other","date"
                "here","how","some","among","can","another","like"
                ,"manner","her","will","somewhat","just","away","near","get")
text <- tm_map(text, removeWords,oth_stopwords)
text <- tm_map(text, stemDocument, language="english")
text <- tm_map(text, removeNumbers)

#Building document term matrix
dtm <- DocumentTermMatrix(text, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.95)
data <- cbind(data, as.matrix(dtm))

# LabelCount Encoding function
labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}

# LabelCount Encoding for Gene and Variation
data$Gene <- labelCountEncoding(data$Gene)
data$Variation <- labelCountEncoding(data$Variation)

# Sentiment analysis
sentiment <- get_nrc_sentiment(data$Text) 
data <- cbind(data,sentiment) 
set.seed(2000)


#To produce sparse matrix
varnames <- setdiff(colnames(data), c("ID", "Class", "Text"))
train_sparse <- Matrix(as.matrix(sapply(data[Class > -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[Class == -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[Class > -1,Class]-1
test_ids <- data[Class == -1,ID]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)
gc() #garbage collection

#Parameters required for xgboost model
parameters <- list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              eta = .2,
              gamma = 1,
              max_depth = 5,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

#Creating foldslist for cv
cv_FoldsList <- createFolds(data$Class[data$Class > -1], k=5, list=TRUE, returnTrain=FALSE)

#Cross-validation and to find optimal number of rounds
xgb_cv <- xgb.cv(data = dtrain,
                 params = parameters,
                 nrounds = 100,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cv_FoldsList,
                 print_every_n = 5,
                 early_stop_round = 23
)
rounds <- which.min(xgb_cv$evaluation_log[,test_mlogloss_mean])
rounds
gc() #garbage collection

#Train the model using XGB
xgb_model <- xgb.train(data = dtrain,
                       params = parameters,
                       watchlist = list(train = dtrain),
                       nrounds = rounds,
                       verbose = 1,
                       print_every_n = 5
)
gc() #garbage collection


#Variable importance: plotting top 24 important variables
names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names,model=xgb_model)
xgb.plot.importance(importance_matrix[1:24])

# Predict the model and thus output the Personalized_submission.csv file
preds <- as.data.table(t(matrix(predict(xgb_model, dtest), nrow=9, ncol=nrow(dtest))))
colnames(preds) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
preds[preds==(apply(preds,1,max))]=1
preds[preds!=apply(preds,1,max)]=0
write.table(data.table(ID=test_ids, preds), "Personalized_submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
#Submission_file=fread("Personalized_submission.csv",sep=",", na.strings = "NA",verbose = FALSE)
