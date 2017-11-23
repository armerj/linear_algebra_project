Notes for Dave's PCA scripts
The training script is the main script, it creates all the eigenvectors, and runs the whole classification loop over all the data, for 1 to 60 eigenvectors.
figure_work.m were some helper script snippets to display figures, run after training.m (training loads all data and results into memory)
final_scores.m was a quick add-on script to read off the scores from the results matrix, for plotting.

Noted For Jon's SVD scripts
testImage is the main script, which takes the path to training and test images, and the type of image. 
getExpressionU get the U matrix for the training data
classifyImage gets the residual for the test image and then determines the minimal value to classify the image. 
