% this helper script is run after training.m, to read off the scores
% in memory
scores_grayscale_test = zeros(1,60);
scores_grayscale_training = zeros(1,60);
scores_sxs_test = zeros(1,60);
scores_sxs_training = zeros(1,60);

for i = 1:60
    scores_grayscale_test(i) = sum(results_pca_test(:,3,i))/76;
    scores_sxs_test(i) = sum(results_pca_test(:,6,i))/76;
    
    scores_grayscale_training(i) = sum(results_pca_training(:,3,i))/683;
    scores_sxs_training(i) = sum(results_pca_training(:,6,i))/683;
end
