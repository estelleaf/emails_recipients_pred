# text_and_graph
FOLDER "DATA"
You will find here the datasets used in the challenge

FOLDER "PREDICTIONS"
You will find there the predictions we made and submitted on the Leaderboard

FOLDER "CODE"
You will find here the algorithms built during the challenge

>> Preprocessing:
Here is a folder with some pre-work
> graphs :
Gives some plots based on graphs
> some_stats:
Some basic statistics 

> template_cross_validation:
A template for all the cross validations we have done.
> paths :
The paths linking to the datasets for each and every participant
> init :
Gives the dictionnaries "adress_books, all_sender..", it describes also the fonction which split the dataset into train/test sets
> loss_function:
Where the loss function is coded

>> Best_submission
This is the folder where we coded the algorithm giving the best submission
> knn
The class and function building the predictor
> submission_knn
Fills the csv file with the predictions, ready to push on the Leaderboard
> find_bestK : 
Grid search to find the best K for our knn

>> Submissions
This folder gathers every other submissions resulting from other algortihms
> submission_centroid :
Fills the csv file with the predictions based on the Centroid algorithm, ready to push on the Leaderboard
tf-idf + centroid
> submission_deeplearning:
Fills the csv file with the predictions based on the neural network algorithm, ready to push on the Leaderboard
> submission_knn_word2vec
Fills the csv file with the predictions based on the Knn algorithm, ready to push on the Leaderboard
word2vec + knn
>submission_mix_knn_rf:
Fills the csv file with the predictions based on an ensemble method mixing random forest and knn, ready to push on the Leaderboard
> sumission_mix_knn_centro_fre:
Fills the csv file with the predictions based on an ensemble method mixing knn, centroid and frequency, ready to push on the Leaderboard
> Random_Forest:
Class for RF


