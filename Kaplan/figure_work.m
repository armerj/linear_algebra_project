%this loads the eigenspaces, test, and training data, if not already
%in workspace


%load('pre_score.mat');
% 
% %display eigenface i from top female grayscale eigs
i = 341;
imagesc(reshape(female_eigs_gray(:,num_female_training-i+1), nrow, ncol));
axis equal;

%display eigenface i from top male grayscale eigs
% i = 342;
% imagesc(reshape(male_eigs_gray(:,num_male_training-i+1), nrow, ncol));
% axis equal;
% 
% %display eigenface i from top female sxs eigs
% i = 2;
% imagesc(reshape(female_eigs_sxs(:,num_female_training-i+1), nrow, ncol*3));
% axis equal;
% 
% %display eigenface i from top male sxs eigs
% i = 3;
% imagesc(reshape(male_eigs_sxs(:,num_male_training-i+1), nrow, ncol*3));
% axis equal;

% number of eigenvectors to use
n = 5;



%create the subsets of eigenvectors to use
female_subset_eigs_gray = female_eigs_gray(:, num_female_training-n+1:num_female_training);
female_subset_eigs_sxs = female_eigs_sxs(:, num_female_training-n+1:num_female_training);

male_subset_eigs_gray = male_eigs_gray(:, num_male_training-n+1:num_male_training);
male_subset_eigs_sxs = male_eigs_sxs(:, num_male_training-n+1:num_male_training);
    
%project test data onto span
Ttest_gray_female = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_female_gray)*female_subset_eigs_gray;
Ttest_gray_male = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_male_gray)*male_subset_eigs_gray;
Ttest_sxs_female = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_female_sxs)*female_subset_eigs_sxs;
Ttest_sxs_male = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_male_sxs)*male_subset_eigs_sxs;

%project training data onto span
Ttrain_gray_female = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_female_gray)*female_subset_eigs_gray;
Ttrain_gray_male = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_male_gray)*male_subset_eigs_gray;
Ttrain_sxs_female = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_female_sxs)*female_subset_eigs_sxs;
Ttrain_sxs_male = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_male_sxs)*male_subset_eigs_sxs;

% individual to project into all the spaces
I = 1;

w_gray_female = Ttest_gray_female(I,:);
w_gray_male = Ttest_gray_male(I,:);

w_sxs_female = Ttest_sxs_female(I,:);
w_sxs_male = Ttest_sxs_male(I,:);

reproject_f_gray = w_gray_female*female_subset_eigs_gray' + avg_female_gray;
reproject_m_gray = w_gray_male*male_subset_eigs_gray' + avg_male_gray;

reproject_f_sxs = w_sxs_female*female_subset_eigs_sxs' + avg_female_sxs;
reproject_m_sxs = w_sxs_male*male_subset_eigs_sxs' + avg_male_sxs;

figure(1)
imagesc(reshape(reproject_f_gray, nrow, ncol));
axis equal;

figure(2)
imagesc(reshape(reproject_m_gray, nrow, ncol));
axis equal;

figure(3)
imagesc(reshape(reproject_f_sxs, nrow, ncol*3));
axis equal;

figure(4)
imagesc(reshape(reproject_m_sxs, nrow, ncol*3));
axis equal;



