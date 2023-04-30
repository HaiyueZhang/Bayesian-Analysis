library(brms)
library(tidyverse)
library(stringr)
library(shinystan)
#library(rstan)
setwd("~/Desktop/Spring 2023/STAT 431/Project")
data = read.csv("nodupe_threeyear.csv")

# 1 Import Data
## Preparing data
data <- mutate(data, Admission.Type = ifelse(grepl("&", Admission.Type), "DiscahrgedAndRecnmitteed", Admission.Type))
data <- mutate(data, IsReturn = ifelse(grepl("New", Admission.Type), 0, 1))
data$Race <- as.factor(data$Race)
data$Sex <- as.factor(data$Sex)
data$Veteran.Status <- as.factor(data$Veteran.Status)
data$Admission.Type <- as.factor(data$Admission.Type)
data$Crime.Class <- as.numeric(data$Crime.Class)
## Delete column Sentence.Years
data <- subset(data, select = -Sentence.Years)
## Rename 
column_index <- which(colnames(data) == "post")
colnames(data)[column_index] <- "IsPost"
column_index1 <- which(colnames(data) == "sentence_yr")
colnames(data)[column_index1] <- "Sentence.Years"
## Standardize(Centralize)
data$Age <- scale(data$Age)
data$Sentence.Years <- scale(data$Sentence.Years)
data

## Split the data set into train and test
set.seed(11237)
n <- dim(data)[1]
fold <- 5
testid <- sample(n, round(n/fold), replace = FALSE)
trainid <- -testid
train <- data[trainid,] # training data for model estimation
test <- data[testid,]

# 2 Train Regression Models

## 2.1 Baseline models

### 2.1.1 Linear Regression Model

#### 2.1.1.1 Main Effect

# Frequentist Main Effect
model_linear_maineffect <- lm(Sentence.Years ~ Age + Sex + IsWhite + Veteran.Status + IsReturn + Crime.Class + IsPost, data = train)
summary(model_linear_maineffect)

#### 2.1.1.2 Interaction
# Frequentist Interaction
model_linear_interaction <- lm(Sentence.Years ~ IsPost * (Age + Sex + IsWhite + Veteran.Status + Crime.Class + IsReturn), data = train)
summary(model_linear_interaction)

#### 2.1.1.3 Maximum_Interaction
# Frequentist two additional Interaction
model_linear_maxinteraction <- lm(Sentence.Years ~ IsPost * Crime.Class * (Age + Sex + IsWhite + Veteran.Status + IsReturn), data = train)
summary(model_linear_maxinteraction)

### 2.1.2 Bayesan Regression model with flat(uninformative) prior and all variables
model_flat <- brm(Sentence.Years ~ Age + Sex + IsWhite + Veteran.Status + IsReturn + Crime.Class + IsPost,
                  data = train,
                  family=brmsfamily("gaussian"),
                  save_all_pars=TRUE,
                  chains = 3,
                  cores = 1,
                  iter = 2000,
                  silent=FALSE)
summary(model_flat)

model_interaction <- brm(Sentence.Years ~ IsPost * (Age + Sex + IsWhite + Veteran.Status + Crime.Class + IsReturn),
                         data = train,
                         family=brmsfamily("gaussian"),
                         save_all_pars=TRUE,
                         chains = 3,
                         cores = 1,
                         iter = 2000,
                         silent=FALSE)
summary(model_interaction)

# Model bayes maxinteraction
model_maxinteraction <- brm(Sentence.Years ~ IsPost * Crime.Class * (Age + Sex + IsWhite + Veteran.Status + IsReturn)
                            data = train,
                            family=brmsfamily("gaussian"),
                            save_all_pars=TRUE,
                            chains = 3,
                            cores = 1,
                            iter = 2000,
                            silent=FALSE)
summary(model_maxinteraction)

### 2.2
model_small <- brm(Sentence.Years ~ Age + Sex + IsWhite + Veteran.Status + Crime.Class + IsPost,
                   data = train,
                   family=brmsfamily("gaussian"),
                   save_all_pars=TRUE,
                   chains = 3,
                   cores = 1,
                   iter = 2000,
                   silent=FALSE)

## 2.3 Model with weakly informative prior

# Since the maximum sentence year is 100 in data cleaning process, it is very unlikely for the absolute value of parameter to exceed 100, thus we use a normal prior with [-100, 100] as 95% interval.
# Fit the model using brm() with custom priors
model_weak <- brm(Sentence.Years ~ Age + Sex + IsWhite + Veteran.Status + Crime.Class + IsPost,
                  data = train,
                  family = brmsfamily("gaussian"),
                  prior = prior(normal(0, 100/1.96), class = 'b'),
                  save_all_pars = TRUE,
                  chains = 3,
                  cores = 1,
                  iter = 1000,
                  silent = FALSE)

## 2.4 Model with horseshoe prior for variable selection

### 2.4.1
# Fit the model using brm() with custom priors
model_horse1 <- brm(Sentence.Years ~ IsPost * (Age + Sex + IsWhite + Veteran.Status + Crime.Class),
                    data = train,
                    family = brmsfamily("gaussian"),
                    prior = set_prior(horseshoe(df = 1)),
                    save_all_pars = TRUE,
                    chains = 3,
                    cores = 1,
                    iter = 1000,
                    silent = FALSE)
summary(model_horse1)

### 2.4.2
# Fit the model using brm() with custom priors
model_horse2 <- brm(Sentence.Years ~ IsPost * Crime.Class * (Age + Sex + IsWhite + Veteran.Status + IsReturn),
                    data = train,
                    family = brmsfamily("gaussian"),
                    prior = set_prior(horseshoe(df = 1)),
                    save_all_pars = TRUE,
                    chains = 3,
                    cores = 1,
                    iter = 1000,
                    silent = FALSE)
summary(model_horse2)

# Diagnosis (Convergence, Diagnosis)
#check trace plot
plot(model_flat)

#check R_hat in summary statistics
summary(model_flat)
summary(model_small)
summary(model_weak)

## Compare test RSS between bayes model and regression model
pred_linear_maineffect  <- predict(object = model_linear_maineffect, newdata=test)
pred_linear_interaction  <- predict(object = model_linear_interaction, newdata=test)
pred_flat <- predict(object = model_flat, newdata=test)
pred_horse1 <- predict(object = model_horse1, newdata=test)
pred_horse2 <- predict(object = model_horse2, newdata=test)
pred_interaction <- predict(object = model_interaction, newdata=test)
pred_maxinteraction <- predict(object = model_maxinteraction, newdata=test)

#residuals = test$Sentence.Years - pred[,1]
rss_linear_maineffect = sum((test$Sentence.Years- pred_linear_maineffect)^2)
rss_linear_interaction = sum((test$Sentence.Years- pred_linear_interaction)^2)
rss_flat = sum((test$Sentence.Years- pred_flat[,1])^2)
rss_horse1 = sum((test$Sentence.Years- pred_horse1[,1])^2)
rss_horse2 = sum((test$Sentence.Years- pred_horse2[,1])^2)
rss_interaction = sum((test$Sentence.Years - pred_interaction[,1])^2)
rss_maxinteraction = sum((test$Sentence.Years - pred_maxinteraction[,1])^2)
rss_linear_maineffect
rss_linear_interaction
rss_flat
rss_horse1
rss_horse2
rss_interaction
rss_maxinteraction

# Model Interpretation

## Visulization
launch_shinystan(model_flat)
launch_shinystan(model_horse2)
plot(conditional_effects(model_flat, effects = "IsPost"))

## Hypothesis testing on all parameters to see which variables have a significant effect
#helper function that can test all parameters
test_all_hypothesis <- function(model, condition) {
  #get all parameter names (only bs)
  var <- colnames(as.data.frame(model$fit))
  var <- var[grepl("b_", var)]
  var <- sub("..", "", var)
  
  hypothesis_res <- as.data.frame(matrix(nrow = 0, ncol = 8))
  for (i in 1:length(var)) {
    hypothesis_str <- paste0(var[i], condition)
    hypothesis_res <- rbind(hypothesis_res, 
                            hypothesis(model,hypothesis_str)$hypothesis)
  }
  hypothesis_res
}

hypothesis_gt0_flat <- test_all_hypothesis(model_horse1, "> 0")
hypothesis_le0_flat <- test_all_hypothesis(model_horse1, "< 0")

## Bayes Factor all parameters to see which variables have a significant effect
#this may crash, reduce loopsize and run separately to avoid crashing
indep_vars <- colnames(model_horse1$data)[-1]
bf_res <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(bf_res) <- c("Variable", "BayesFactor", "Direction")

for (i in 1:length(indep_vars)) {
  formula_less <- as.formula(paste("~ . - ", as.name(indep_vars[i])))
  model_less <- update(model_flat, formula. = formula_less)
  bf <- bayes_factor(model_flat, model_less)
  bf <- as.character(bf)
  new_row <- data.frame(Variable = indep_vars[i], BayesFactor = bf[1], DIrection = bf[2])
  bf_res <- rbind(bf_res, new_row)
}
bf_res

#helper function that can test all parameters
test_all_hypothesis <- function(model, condition) {
  #get all parameter names (only bs)
  var <- colnames(as.data.frame(model$fit))
  var <- var[grepl("b_", var)]
  var <- sub("..", "", var)
  
  hypothesis_res <- as.data.frame(matrix(nrow = 0, ncol = 8))
  for (i in 1:length(var)) {
    hypothesis_str <- paste0(var[i], condition)
    hypothesis_res <- rbind(hypothesis_res, 
                            hypothesis(model,hypothesis_str)$hypothesis)
  }
  hypothesis_res
}

hypothesis_gt0_flat <- test_all_hypothesis(model_horse2, "> 0")
hypothesis_le0_flat <- test_all_hypothesis(model_horse2, "< 0")
indep_vars <- colnames(model_horse2$data)[-1]
bf_res <- data.frame(matrix(ncol = 3, nrow = 0))
colnames(bf_res) <- c("Variable", "BayesFactor", "Direction")

for (i in 1:length(indep_vars)) {
  formula_less <- as.formula(paste("~ . - ", as.name(indep_vars[i])))
  model_less <- update(model_flat, formula. = formula_less)
  bf <- bayes_factor(model_flat, model_less)
  bf <- as.character(bf)
  new_row <- data.frame(Variable = indep_vars[i], BayesFactor = bf[1], DIrection = bf[2])
  bf_res <- rbind(bf_res, new_row)
}
bf_res