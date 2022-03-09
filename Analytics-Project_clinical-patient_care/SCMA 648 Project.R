setwd('C:\\Users\\Dell\\PG\\MS\\Sem 1\\BDA\\Project')

# Importing the necessary packages
library(dplyr)
library(dummies)
library(rgl)
library(cluster)
library(fpc)
library(rpart)
library(ROCR)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(ggplot2)

# Reading the file
col_names <-scan("model_data.csv", what="character", nlines=1, sep=",")
project_data=read.table("model_data.csv", 
                        header=TRUE,  
                        sep=",",
                        colClasses = c("character", "numeric",rep("factor",4), 
                                       "numeric", "factor", rep("numeric", 7)))

save(project_data, file="project_data.rda")
load("project_data.rda")

head(project_data)

set.seed(12345)

# Selecting the required columns
my_df <- project_data %>%
  dplyr::select(BloodPressureUpper,DRG01, BloodPressureLower,Gender,Age, 
                PrimaryInsuranceCategory,BloodPressureDiff,Pulse,PulseOximetry,
                Respirations,Temperature, OU_LOS_hrs)

summary(my_df)

# Outlier Detection
boxplot(my_df$BloodPressureUpper)
boxplot(my_df$BloodPressureLower)
boxplot(my_df$Pulse)
boxplot(my_df$BloodPressureDiff)
boxplot(my_df$Respirations)

# Outlier Removal
df <- my_df %>%
  dplyr::filter(BloodPressureUpper < 210 &
                 BloodPressureLower < 110 &
                 Pulse < 130 &
                 BloodPressureDiff > 15 & BloodPressureDiff < 100 &
                 Respirations > 12 & Respirations < 23
                
                ) %>%
  droplevels()


summary(df)

class(df$DRG01)

# Imputation
df$Temperature[is.na(df$Temperature)] <- median(df$Temperature, na.rm=TRUE)
summary(df)

sapply(df, class)

# Dummy Variables
df_new <- dummy.data.frame(df, names=c("Gender", "PrimaryInsuranceCategory", "DRG01"))

summary(df_new)

# Scaling
my_df_scale <- scale(df_new, center=TRUE, scale=TRUE)

# K-Means Clustering
my_kmeans <- kmeans(my_df_scale, centers=3)

set.seed(12345)

#PCA
my_pca <- prcomp(my_df_scale, retx=TRUE)
plot(my_pca$x[,1:2], col=my_kmeans$cluster, pch=my_kmeans$cluster)
legend("topright", legend=1:3, col=1:3, pch=1:3)

plot(my_pca$x[,1:2], type="n")
points(my_pca$x[my_kmeans$cluster == 1,1:2], 
       col=my_kmeans$cluster[my_kmeans$cluster== 1], 
       pch=my_kmeans$cluster[my_kmeans$cluster == 1])

plot(my_pca$x[,1:2], type="n")
points(my_pca$x[my_kmeans$cluster == 2,1:2], 
       col=my_kmeans$cluster[my_kmeans$cluster== 2], 
       pch=my_kmeans$cluster[my_kmeans$cluster == 2])

plot(my_pca$x[,1:2], type="n")
points(my_pca$x[my_kmeans$cluster == 3,1:2], 
       col=my_kmeans$cluster[my_kmeans$cluster== 3], 
       pch=my_kmeans$cluster[my_kmeans$cluster == 3])

my_pca$rotation[,1:2]

#*******************************************************************************

# Models

model_df <- project_data %>%
  dplyr::select(BloodPressureUpper,DRG01, BloodPressureLower,Gender,Age, 
                PrimaryInsuranceCategory,BloodPressureDiff,Pulse,PulseOximetry,
                Respirations,Temperature,Flipped)
head(model_df)

# Outlier Removal
model_df <- model_df %>%
  dplyr::filter(BloodPressureUpper < 210 &
                  BloodPressureLower < 110 &
                  Pulse < 130 &
                  BloodPressureDiff > 15 & BloodPressureDiff < 110 &
                  Respirations > 12 & Respirations < 23
  ) %>%
  droplevels()


summary(model_df)

# Imputation
model_df$Temperature[is.na(model_df$Temperature)] <- median(model_df$Temperature, na.rm=TRUE)
summary(model_df)

sapply(model_df, class)

# Train-test split
train_rows <- createDataPartition(model_df$Flipped, p=0.7, list=FALSE)

train_1 <- model_df[train_rows,]
test_1 <- model_df[-train_rows,]

summary(train_1$Flipped)
summary(test_1$Flipped)

# LOGISTIC REGRESSION

lr <- glm(Flipped ~ ., data=train_1, family=binomial("logit"))

summary(train_1)
summary(test_1)

lr_predict <- predict(lr, newdata=test_1, type="response")
lr_predict_class <- character(length(lr_predict))
lr_predict_class[lr_predict < 0.5] <- "0"
lr_predict_class[lr_predict >= 0.5] <- "1"

# Confusion Matrix
cm_1 = table(test_1$Flipped, lr_predict_class)
cm_1

# Important Variables
summary(lr)

# Misclassification Rate
1-sum(diag(cm_1))/sum(cm_1)

train_1 <- train_1 %>%
   mutate(Flipped =
            factor(if_else(Flipped == "0", "DNF", "F"),
                   levels=c("F", "DNF"))) %>%
   droplevels()

test_1 <- test_1 %>%
   mutate(Flipped =
            factor(if_else(Flipped == "0", "DNF", "F"),
                   levels=c("F", "DNF"))) %>%
   droplevels()

# DECISION TREE

rpart_1 <- rpart(Flipped ~ ., data=train_1)
rpart_predict <- predict(rpart_1, newdata=test_1, type="class")

# Confusion Matrix
cm_2 = table(test_1$Flipped, rpart_predict)
cm_2

# Important Variables
rpart_1$variable.importance

# Misclassification Rate
1-sum(diag(cm_2))/sum(cm_2)

# RANDOM FOREST

table(train_1$Flipped)
rf <- randomForest(Flipped ~ .,data = train_1,importance=TRUE)
rf$importance

predict_rf <- predict(rf, newdata=test_1, type="class")

# Confusion Matrix
cm_3 = table(test_1$Flipped, predict_rf)
cm_3

# Important Variables
rf$importance

# Misclassification Rate
1-sum(diag(cm_3))/sum(cm_3)

# SVM - Linear Weights

# Dummy Variables
train <- dummy.data.frame(train_1, names=c("Gender", "PrimaryInsuranceCategory", "DRG01"))
preprocess <- preProcess(train)
train_numeric <- predict(preprocess, train)

test <- dummy.data.frame(test_1, names=c("Gender", "PrimaryInsuranceCategory", "DRG01"))
test_numeric <- predict(preprocess, test)

svm <- train(Flipped ~ .,data=train_numeric,method="svmLinearWeights",metric="ROC",
                      trControl=trainControl(classProbs=TRUE,
                                             summaryFunction=twoClassSummary))
svm

modelLookup("svmLinearWeights")

predict_svm <- predict(svm, newdata=test_numeric)

# Confusion Matrix
cm_4 <- table(test_numeric$Flipped, predict_svm)
cm_4

# Misclassification Rate
1-sum(diag(cm_4))/sum(cm_4)

# SVM - Radial Weights

svm_rbf <- train(Flipped ~ .,data=train_numeric,method="svmRadialWeights",metric="ROC",
                          trControl=trainControl(classProbs=TRUE,
                                                 summaryFunction=twoClassSummary))

predict_svm_rbf <- predict(svm_rbf, newdata=test_numeric)

# Confusion Matrix
cm_5 <- table(test_numeric$Flipped, predict_svm_rbf)
cm_5

# Misclassification Rate
1-sum(diag(cm_5))/sum(cm_5)

# Measuring performance
pred1 <- predict(lr, test_1, type="response")
lrpred <- prediction(pred1,test_1$Flipped, label.ordering=c("DNF", "F"))
lr_perf <- performance(lrpred, "tpr", "fpr")

pred2 <- predict(rpart_1, test_1, type="prob")
rpartpred <- prediction(pred2[,1],test_1$Flipped,label.ordering=c("DNF", "F"))
rpart_perf <- performance(rpartpred, "tpr", "fpr")

pred3 <- predict(rf, newdata=test_1, type="prob")
rfpred <- prediction(pred3[,1],test_1$Flipped,label.ordering=c("DNF", "F"))
rf_perf <- performance(rfpred, "tpr", "fpr")

pred4 <- predict(svm, newdata=test_numeric, type="prob")
svmpred <- prediction(pred4[,1],test_numeric$Flipped,label.ordering=c("DNF", "F"))
svm_perf <- performance(svmpred, "tpr", "fpr")

pred5 <- predict(svm_rbf, newdata=test_numeric, type="prob")
svmrbfpred <- prediction(pred5[,1],test_numeric$Flipped,label.ordering=c("DNF", "F"))
svm_rbf_perf <- performance(svmrbfpred, "tpr", "fpr")

# ROC Curves
plot(lr_perf, col=1)
plot(rpart_perf, col=2, add=TRUE)
plot(rf_perf, col=3, add=TRUE)
plot(svm_perf, col=4, add=TRUE)
plot(svm_rbf_perf, col=5,add=TRUE)
legend("bottomright",c("LR", "CT","RF","SVM linear", "SVM RBF"), col=1:5, lwd=0.5, cex = 0.7)

# AUC for the models
lr_auc <- performance(lrpred, "auc")
lr_auc@y.values[[1]]

rpart_auc <- performance(rpartpred, "auc")
rpart_auc@y.values[[1]]

rf_auc <- performance(rfpred, "auc")
rf_auc@y.values[[1]]

svm_auc <- performance(svmpred, "auc")
svm_auc@y.values[[1]]

svmrbf_auc <- performance(svmrbfpred, "auc")
svmrbf_auc@y.values[[1]]

# The variable gender plays an important role in determining whether the patient flipped or not
mp <- table(model_df$Gender, model_df$Flipped)
mosaicplot(mp, color=c(1:2), xlab="Gender",
           ylab="Flipped",main="Gender v/s Flipped")

bp <- table(model_df$Flipped, model_df$Gender)
barplot(bp, main = "Flipped according to gender", xlab = "Gender",col = c("red","green"), beside = TRUE)
legend("topright", c("Not Flipped","Flipped"),fill = c("red","green"),cex = 0.7)

# The variable DRG01 plays an important role in determining whether the patient flipped or not
mp2 <- table(model_df$DRG01, model_df$Flipped)
mosaicplot(mp2, color=c(1:2), xlab="Diagnostic Code",
           ylab="Flipped",main="Diagnostic Code v/s Flipped")

# SVM RBF gives the highest AUC for the data
# Therefore, we will SVM RBF for prediction in the prediction data. 

# Prediction data

col_names1 <-scan("prediction_data.csv", what="character", nlines=1, sep=",")
pred_data=read.table("prediction_data.csv", 
                        header=TRUE,  
                        sep=",",
                        colClasses = c("character", "numeric",rep("factor",3), 
                                        rep("numeric", 7)))

save(pred_data, file="pred_data.rda")
load("pred_data.rda")

head(pred_data)

set.seed(12345)

pred_df <- pred_data %>%
  dplyr::select(BloodPressureUpper,DRG01, BloodPressureLower,Gender,Age, 
                PrimaryInsuranceCategory,BloodPressureDiff,Pulse,PulseOximetry,
                Respirations,Temperature)

summary(pred_df)

# Imputation
pred_df$Temperature[is.na(pred_df$Temperature)] <- median(model_df$Temperature, na.rm=TRUE)
pred_df$PulseOximetry[is.na(pred_df$PulseOximetry)] <- median(model_df$PulseOximetry, na.rm=TRUE)
pred_df$BloodPressureDiff[is.na(pred_df$BloodPressureDiff)] <- median(model_df$BloodPressureDiff, na.rm=TRUE)
pred_df$Pulse[is.na(pred_df$Pulse)] <- median(model_df$Pulse, na.rm=TRUE)
pred_df$Respirations[is.na(pred_df$Respirations)] <- median(model_df$Respirations, na.rm=TRUE)
pred_df$BloodPressureUpper[is.na(pred_df$BloodPressureUpper)] <- median(model_df$BloodPressureUpper, na.rm=TRUE)

summary(pred_df)

# Pre-processing
p <- dummy.data.frame(pred_df, names=c("Gender", "PrimaryInsuranceCategory", "DRG01"))
p1 <- predict(preprocess, p)

# Prediction
predictions <- predict(svm_rbf, newdata=p1, type = "prob")
predictions

preds <- data.frame(Obkey = pred_data$ObservationRecordKey,pred = predictions)
preds <- preds[ -c(3) ]
preds$pred.F <- round(preds$pred.F, digit = 2)
preds

# Saving the predictions
write.csv(data.frame(preds), "predictions.csv", row.names = FALSE)

#*******************************************************************************