## Load the edx and validation data using the scrpit provided below

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
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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


## Explore edx data set

# display edx in tibble form, to see some characteristics of the data strucrure 
edx %>% as_tibble()

#Number of unique users and movies in the edx data set
edx %>% summarize(users = n_distinct(userId), movies = n_distinct(movieId))

#Make sure no NA are present in any column of edx.
colSums(is.na(edx))

#explore movie's effect on rating by ploting the number of ratings per movie and
# the average rating per movie.
edx %>% 
  group_by(movieId) %>% summarize(n=n()) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  xlab("Number of ratings") +
  ylab("Number of movies") 

edx %>% 
  group_by(movieId) %>% summarize(rating=mean(rating)) %>% 
  ggplot(aes(rating)) +  
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  xlab("Ratings") +
  ylab("Number of ratings") 

# explore user's effect on the rating by ploting the number of ratings per user 
# and the average user's rating per user.
edx %>% 
  group_by(userId) %>% summarize(n=n()) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  xlab("Number of ratings") +
  ylab("Number of users") 

edx %>% 
  group_by(userId) %>% summarize(rating=mean(rating)) %>% 
  ggplot(aes(rating)) +  
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() + 
  xlab("Ratings") +
  ylab("Number of users") 

# explore the actual ratings made by the users 
edx %>% 
  group_by(rating) %>% summarize(n=n()) %>% 
  ggplot(aes(rating,n)) + geom_bar(stat = "identity", fill="black") +
  scale_x_log10() + 
  xlab("Ratings") +
  ylab("Number of Ratings")

# The overall mean rating is
mean(edx$rating)

# The move release year is given as part of the title - so let's extract the 
# release year from the movie title and put it in a spearate column called year
edx<-edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# explore the movie release year's effect on the rating by ploting the number of 
# ratings and average rating per movie release year.
edx %>% 
  group_by(year) %>% 
  summarize(n=n()) %>%
  ggplot(aes(year,n)) + 
  geom_bar(stat = "identity", fill="black") +
  scale_x_log10() + 
  xlab("Release Year") +
  ylab("Number of Ratings")

edx %>% 
  group_by(year) %>% 
  summarize(avg=mean(rating)) %>%
  ggplot(aes(year,avg)) + geom_smooth() +
  scale_x_log10() + 
  xlab("Release Year") +
  ylab("Average rating for the year")

# explore the genres's effect on the rating by ploting the number of the ratings 
# per genres. But first, to be able to explore of the individual genres's effect, 
# let's split the composite genres into individual genres.                                  
edx_genres<- edx %>%  separate_rows(genres, sep = "\\|") 
edx_genres %>%
  group_by(genres) %>%
  summarise(n=n()) %>%
  ggplot(aes(genres,n)) + 
  geom_bar(stat = "identity", fill="black") +
  theme(axis.text.x = element_text(angle = 60, hjust = 1))+  
  xlab("Genres") +
  ylab("Number of Ratings")

#explore rating time effect on the rating by rounding the timestamp into rating 
#week and plot the number of the ratings per week.
edx_time <- edx %>% mutate(week = round_date(as_datetime(timestamp), unit = "week"))
edx_time %>% 
  group_by(week) %>%
  summarize(rating=mean(rating)) %>%
  ggplot(aes(week, rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Average Ratings") +
  ylab("Rating time (week)")

# let's start building different models and check the RMSE results. # But first,
# let's partition edx into two parts: edx_train (80%) and edx_test (20%). edx_train 
# will be used to build/train the model, while edx_test is used for cross-validating the model.  

set.seed(2)
edx_test_index <- createDataPartition(y = edx$rating, times = 1,
                                      p = 0.2, list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_test_tmp <- edx[edx_test_index,]

# get all records in edx_test_tmp with corresponding records in 
# edx_train and set it to edx_test data set 
edx_test <- edx_test_tmp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# remove those records in edx_test_tmp without corresponding records 
#in edx_train, and put them back to edx_train. 
removed_tmp <- anti_join(edx_test_tmp, edx_test)
edx_train <- rbind(edx_train, removed_tmp)

# the following function will be used to calculate the RMSE, for a 
# given true_ratings and predicted_ratings, throughout the script.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# the following would be used to store the RMSE values of the different model 
rmse_results <- tibble(Method=character(), RMSE=double()) 

# "Overall average" model - use the overall average rating as prediction to 
# every user/movie combinations.

# get the overall average value, mu.
mu<-mean(edx_train$rating)

# Calculate the RMSE for the above "Overall average" model and store the 
# result in the rmse_results table.
rmse<-RMSE(edx_test$rating, mu)
rmse_results <- bind_rows(rmse_results,tibble(Method="Overall average", RMSE=rmse))
rmse_results %>% knitr::kable()

# "Overall average + Movie" Model - add the movie effect to the "Overall average" model.
mu <- mean(edx_train$rating)
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by = "movieId") %>% 
  mutate(pred = mu + b_i) %>% .$pred

# Calculate the RMSE for the above "Overall average + Movie" model and store the result 
# in the rmse_results table.   
rmse<-RMSE(predicted_ratings, edx_test$rating) 
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Overall average + Movie", RMSE=rmse))
rmse_results %>% knitr::kable()

# regularize the prediction to get better result - penalize large estimates that are 
# based on small sample sizes. Test the model with different values of lambda to get 
# the lambda value that minimize the RMSE.    
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating) 
  movie_avgs <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  
  predicted_ratings <- edx_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    mutate(pred = mu + b_m) %>% .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})
# Take the lowest RMSE and store the result in the results table.
rmse_results <- bind_rows(rmse_results, 
                          tibble(Method="Overall average + Movie + Regularize", RMSE=min(rmses)))
rmse_results %>% knitr::kable()

# add the user's effect to the model.
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating) 
  movie_avgs <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  user_avgs <- edx_train %>%
    left_join(movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating -mu -b_i)/(n()+l))
  
  predicted_ratings <- edx_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>% .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Overall average + Movie + User + Regularize", RMSE=min(rmses)))
rmse_results %>% knitr::kable()

# add movie release year effect
lambdas <-seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_train$rating) 
  movie_avgs <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  user_avgs <- edx_train %>%
    left_join(movie_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating -mu -b_i)/(n()+l))
  year_avgs <- edx_train %>%
    left_join(movie_avgs, by="movieId") %>%
    left_join(user_avgs, by="userId") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating -mu -b_i -b_u)/(n()+l))
  
  predicted_ratings <- edx_test %>%
    left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(year_avgs, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_y) %>% .$pred
  return(RMSE(predicted_ratings, edx_test$rating))
})
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Overall average + Movie + User + year + Regularize", RMSE=min(rmses)))
rmse_results %>% knitr::kable()

#Get the lambda that results in minimum RMSE, and use it for the final model.
lambda <- lambdas[which.min(rmses)]
lambda

# Final model - "Overall average + Movie + User + year + Regularize" with lambda=4.75  
# we will use the whole edx data set to train the final model and use the validation 
# data set it and get the final RMSE value.

# since we extract the movie release year from the edx data set and used it for 
# predication, we will need to extract the movie release years from the validation data too.
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation %>% as_tibble()

# Make sure userId and movieId in validation set are also in edx set
#(i.e. to make sure we have a valid prediction when using validation data set)
validation_exist_in_edx <- validation %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
identical(validation_exist_in_edx,validation)

mu <- mean(edx$rating) 
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating -mu -b_i)/(n()+lambda))
year_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating -mu -b_i -b_u)/(n()+lambda))

predicted_ratings <-
  validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(year_avgs, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_y) %>% .$pred

Final_RMSE<-(RMSE(predicted_ratings, validation$rating))

rmse_results <- bind_rows(rmse_results, tibble(Method="Final model", RMSE=Final_RMSE))
rmse_results %>% knitr::kable()
