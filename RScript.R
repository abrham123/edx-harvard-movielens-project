## Abrham Birru
## MovieLens Project 
## HarvardX: PH125.9x - Capstone Project
## https://github.com/abrham123/

#################################################
# MovieLens Rating Project Code 
################################################

## Introduction
# This data analysis report is prepared as a part of HarvardX Data Science Capstone (HarvardX: PH125.9x) Project.The given dataset(movielens) has been analysed by using different tools and techniques. 

# In this project, I will combine several machine learning strategies to construct a movie recommendation system based on the "MovieLens" dataset.

## Goal
#  The goal is to predict movie ratings, and evaluate the accuracy of the predicted model from the given code the dataset called  "edx" was split into the training and validation sets.

## Data Loading
# The code is provided in the edx capstone project module:
# The following lines of code will create training and validation sets

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


## Exploring The Data

head(edx)
head(validation)
str(edx)
str(validation)
names(edx)
names(validation)

# summary of unique movies and users
summary(edx)

# Number of unique movies and users in dataset 
edx %>%
  summarize(number_of_users = n_distinct(userId), 
            number_of_movies = n_distinct(movieId))

# Plot Rating distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

# Make a plot of number of ratings per movie in edx dataset
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

# only once rated 20 movies table
edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Make a plot of mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "blue") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

# Make a plot of number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

### Modelling

## Model-1

# Let's take the average(mean)
# This model is developed by using average rating (mean) to train the model and predict the movie rating in the validation model.
average <- mean(edx$rating)
average

# Initiate RMSE and store based on simple prediction
naive_rmse <- RMSE(validation$rating, average)
naive_rmse

rmse_data <- data_frame(method = "Average Movie Rating Model", RMSE = naive_rmse)
rmse_data %>% knitr::kable()

## Model-2 ##

# In this model the Movies are used to examine the predictive efficiency
# Simple model taking into account the movie effect b_i
# Subtract the rating minus the mean for each rating the movie received
# Plot number of movies with the computed b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - average))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("blue"),
                     ylab = "Number of movies", main = "Number of movies with the computed b_i")


# Store rmse data results 
predicted_ratings <- average +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_data <- bind_rows(rmse_data,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse ))
# Finally Check rmse data results
rmse_data %>% knitr::kable()

## Model-3
# Here we will model the Movie-User effects
user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - average - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("blue"))

# User averages group by user id
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - average - b_i))


# Test and save rmse data results 
predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = average + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_data <- bind_rows(rmse_data,
                          data_frame(method="Movie-User Effect Model",  
                                     RMSE = model_2_rmse))

# Check rmse data result
rmse_data %>% knitr::kable()

## Model-4
# Here Cross validation is used to select optimum value of lambda.It is a regularized model.

lambdas <- seq(0, 10, 0.25)


# For each lambda,find b_i & b_u, and followed by rating prediction and testing
rmses <- sapply(lambdas, function(l){
  average <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - average)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - average)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = average + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


# Make a Plot rmses vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  


# The optimal lambda                                                             
lambda <- lambdas[which.min(rmses)]
lambda

# Test and save results                                                             
rmse_data <- bind_rows(rmse_data,
                          data_frame(method="Regularized Movie and User Effect Model",  
                                     RMSE = min(rmses)))

# Result                                                         
rmse_data %>% knitr::kable()

## Appendix
print("Operating System:")
version

