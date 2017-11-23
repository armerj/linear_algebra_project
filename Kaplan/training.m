%work in progress
%by Dave Kaplan


num_female_training = 341;
num_male_training = 342;

num_female_test = 38;
num_male_test = 38;

nrow = 200;
ncol = 180;



% Stack the images, male and female, in flattened row vector vectors into a 2d array
% The first num_female_training row or num_female_test rows are female
% followed by num_male_training rows or num_male test rows
%
training_data_gray = zeros(num_female_training + num_male_training, nrow*ncol);
training_data_sxs = zeros(num_female_training + num_male_training, nrow*ncol*3);

test_data_gray = zeros( num_female_test + num_male_test, nrow*ncol);
test_data_sxs = zeros(num_female_test + num_male_test, nrow*ncol*3);

divisor = 255.0*ones(1,nrow*ncol);
divisor2 = 255.0*ones(1,nrow*ncol*3);

 % load the training examples
 for i = 1:(num_female_training+num_male_training)
     filename = cat(2,num2str(i), '_imgray.bmp');
     filename = cat(2,'all_training/', filename);
     im =  imread(filename);
     data = double((reshape(im,1, nrow*ncol)))./divisor;
     training_data_gray(i,:) = data;
 end
 
  % load the test examples
 for i = 1:(num_female_test+num_male_test)
     filename = cat(2,num2str(i), '_imgray.bmp');
     filename = cat(2,'all_test/', filename);
     im =  imread(filename);
     data = double((reshape(im,1, nrow*ncol)))./divisor;
     test_data_gray(i,:) = data;
 end
 
 clear im;
%  %this will display image 1: imagesc(reshape(training_data(1,:), nrow, ncol))


 
 
%this section creates the 3 stacked side by side images, flattened 
%to row vectors, all in one array, for the training data
for i = 1:(num_female_training+num_male_training)
    filename = cat(2, num2str(i), '.jpg');
    filename = cat(2, 'all_training/', filename);
    im =  imread(filename);
    
    R = im(:,:,1); 
    G = im(:,:,2);
    B = im(:,:,3);


    sxs = zeros(nrow, 3*ncol);
    sxs(:,1:ncol) = R;
    sxs(:,ncol+1:2*ncol) = G;
    sxs(:,2*ncol+1:3*ncol) = B;
    
    data2 = double((reshape(sxs,1, nrow*ncol*3)))./divisor2;
    training_data_sxs(i,:) = data2;
end
clear R G B sxs;

%this section creates the 3 stacked side by side images, flattened 
%to row vectors, all in one array, for the test data
for i = 1:(num_female_test+num_male_test)
    filename = cat(2, num2str(i), '.jpg');
    filename = cat(2, 'all_test/', filename);
    im =  imread(filename);
    
    R = im(:,:,1); 
    G = im(:,:,2);
    B = im(:,:,3);


    sxs = zeros(nrow, 3*ncol);
    sxs(:,1:ncol) = R;
    sxs(:,ncol+1:2*ncol) = G;
    sxs(:,2*ncol+1:3*ncol) = B;
    
    data2 = double((reshape(sxs,1, nrow*ncol*3)))./divisor2;
    test_data_sxs(i,:) = data2;
end
clear R G B sxs;


%create the average face for the gray_scale and sxs training images
avg_face_gray = mean(training_data_gray, 1);
avg_female_gray = mean(training_data_gray(1:num_female_training,:));
avg_male_gray = mean(training_data_gray(num_female_training+1:num_female_training+num_male_training,:));



avg_face_sxs = mean(training_data_sxs, 1);
avg_female_sxs = mean(training_data_sxs(1:num_female_training,:));
avg_male_sxs = mean(training_data_sxs(num_female_training+1:num_female_training+num_male_training,:));


% ***************************************************************
% create the female eigenspace for grayscale
%subtract out the average female from all the female rows,
%store into F_gray matrix
F_gray = training_data_gray(1:num_female_training,:)-ones(num_female_training,1) * avg_female_gray;


%A is only num_female_training * num_female_training so easy to compute with
A = F_gray * F_gray';
[U,~] = eig(A); %the tilde represents the unused eigenvalues

% get female eigenface for the grayscale training images
V = F_gray'*U; %this replacement for U would select less: U(:,num_female_training-num_eigs+1:num_female_training)

%normalize eigenvectors
normsV=diag(sqrt(diag(V' * V)));
female_eigs_gray = V * normsV^(-1);
clear A F_gray U V normsV;
% **************************************************************


% **************************************************************
% create the female eigenspace for RGB sxs
F_sxs = training_data_sxs(1:num_female_training,:)-ones(num_female_training,1) * avg_female_sxs;
A = F_sxs * F_sxs';
[U,~] = eig(A);

% create eigenfaces for female sxs
V = F_sxs'*U;

%normalize eigenvectors
normsV=diag(sqrt(diag(V' * V)));
female_eigs_sxs = V * normsV^(-1);
clear A F_sxs U V normsV;

%create the male eigenspace for grayscale
M_gray = training_data_gray(num_female_training+1:num_female_training+num_male_training,:)-ones(num_male_training,1) * avg_male_gray;
A = M_gray * M_gray';
[U,~] = eig(A); 
% male grayscale eigenfaces
V = M_gray'*U;

%normalize eigenvectors
normsV=diag(sqrt(diag(V' * V)));
male_eigs_gray = V * normsV^(-1);
clear A M_gray U V normsV;

%create the male eigenspace for sxs
M_sxs = training_data_sxs(num_female_training+1:num_female_training+num_male_training,:)-ones(num_male_training,1) * avg_male_sxs;
A = M_sxs * M_sxs';
[U,~] = eig(A);
% eigenfaces for male sxs
V = M_sxs'*U;

%normalize eigenvectors
normsV=diag(sqrt(diag(V' * V)));
male_eigs_sxs = V * normsV^(-1);
clear A M_sxs U V normsV;

% % ***********************************
% % this section for displaying figures
% figure(1);
% imagesc(reshape(avg_face_gray, nrow, ncol));
% figure(2);
% imagesc(reshape(avg_female_gray, nrow, ncol));
% figure(3);
% imagesc(reshape(avg_male_gray, nrow, ncol));
% 
% figure(4);
% imagesc(reshape(avg_face_sxs, nrow, ncol*3));
% figure(5);
% imagesc(reshape(avg_female_sxs, nrow, ncol*3));
% figure(6);
% imagesc(reshape(avg_male_sxs, nrow, ncol*3));
% 
% figure(7);
% imagesc(reshape(female_eigs_gray(:,num_female_training), nrow, ncol));
% axis equal;
% 
% figure(8);
% imagesc(reshape(female_eigs_gray(:,num_female_training-1), nrow, ncol));
% axis equal;
% 
% figure(9);
% imagesc(reshape(female_eigs_gray(:,num_female_training-2), nrow, ncol));
% axis equal;
% 
% figure(10);
% imagesc(reshape(female_eigs_gray(:,num_female_training-3), nrow, ncol));
% axis equal;
% 
% figure(11);
% imagesc(reshape(female_eigs_gray(:,num_female_training-4), nrow, ncol));
% axis equal;
% 
% figure(12);
% imagesc(reshape(female_eigs_sxs(:,num_female_training), nrow, ncol*3));
% axis equal;
% 
% figure(13);
% imagesc(reshape(female_eigs_sxs(:,num_female_training-1), nrow, ncol*3));
% axis equal;
% 
% figure(14);
% imagesc(reshape(female_eigs_sxs(:,num_female_training-2), nrow, ncol*3));
% axis equal;
% 
% figure(15);
% imagesc(reshape(female_eigs_sxs(:,num_female_training-3), nrow, ncol*3));
% axis equal;
% 
% figure(16);
% imagesc(reshape(female_eigs_sxs(:,num_female_training-4), nrow, ncol*3));
% axis equal;
% 
% figure(17);
% imagesc(reshape(male_eigs_gray(:,num_male_training), nrow, ncol));
% axis equal;
% 
% figure(18);
% imagesc(reshape(male_eigs_gray(:,num_male_training-1), nrow, ncol));
% axis equal;
% 
% figure(19);
% imagesc(reshape(male_eigs_gray(:,num_male_training-2), nrow, ncol));
% axis equal;
% 
% figure(20);
% imagesc(reshape(male_eigs_gray(:,num_male_training-3), nrow, ncol));
% axis equal;
% 
% figure(21);
% imagesc(reshape(male_eigs_gray(:,num_male_training-4), nrow, ncol));
% axis equal;
% 
% figure(22);
% imagesc(reshape(male_eigs_sxs(:,num_male_training), nrow, ncol*3));
% axis equal;
% 
% figure(23);
% norm(test_data_gray(i,:)-avg_female_gray
% axis equal;
% 
% figure(24);
% imagesc(reshape(male_eigs_sxs(:,num_male_training-2), nrow, ncol*3));
% axis equal;
% 
% figure(25);
% imagesc(reshape(male_eigs_sxs(:,num_male_training-3), nrow, ncol*3));
% axis equal;
% 
% figure(26);
% imagesc(reshape(male_eigs_sxs(:,num_male_training-4), nrow, ncol*3));
% axis equal;


% ***************************************
% CLASSIFIER BELOW THIS LINE

% Baseline classifier results matrix
% col 1 = euclidean distance of the example from the average female grayscale
% col 2 = euclidean distance of the example from the average male grayscale
% col 3 = score euclidean distance grayscale, 1.0 for correct, 0 for
% incorrect
% col 3 = euclidean distance from avg female RGB side by side
% col 4 = euclidean distance from avg male RGB side by side
% col 6 = score euclidean distance RBG sxs, 1.0 for correct, 0 for
% incorrect

results_euclidean_test = zeros(num_female_test + num_male_test, 6);
results_euclidean_training = zeros(num_female_training + num_male_training, 6);

% test the test data euclidean distance
for i = 1:num_female_test+num_male_test
    results_euclidean_test(i,1) = norm(test_data_gray(i,:)-avg_female_gray);
    results_euclidean_test(i,2) = norm(test_data_gray(i,:)-avg_male_gray);
    
    results_euclidean_test(i,4) = norm(test_data_sxs(i,:)-avg_female_sxs);
    results_euclidean_test(i,5) = norm(test_data_sxs(i,:)-avg_male_sxs);
    
    if (i < 39)
        % the test example is female
        
        if (results_euclidean_test(i,1) < results_euclidean_test(i,2))
            % distance from average female is less than distance from
            % average male
            results_euclidean_test(i,3) = 1.0;
        end
        if (results_euclidean_test(i,4) < results_euclidean_test(i,5))
            % distance from average female is less than distance from
            % average male
            results_euclidean_test(i,6) = 1.0;
        end
    else
        % the test example is male
        
        if (results_euclidean_test(i,2) <= results_euclidean_test(i,1))
            % distance from average male is less than or equal to
            % than distance from the average female
            results_euclidean_test(i,3) = 1.0;
        end
        if (results_euclidean_test(i,5) <= results_euclidean_test(i,4))
            % distance from average male is less than or equal to
            % than distance from the average female
            results_euclidean_test(i,6) = 1.0;
        end 
       
    end
    
end

% test the training data euclidean distance
for i = 1:num_female_training+num_male_training
    results_euclidean_training(i,1) = norm(training_data_gray(i,:)-avg_female_gray);
    results_euclidean_training(i,2) = norm(training_data_gray(i,:)-avg_male_gray);
    
    results_euclidean_training(i,4) = norm(training_data_sxs(i,:)-avg_female_sxs);
    results_euclidean_training(i,5) = norm(training_data_sxs(i,:)-avg_male_sxs);
    
    if (i <= num_female_training)
        % the test example is female
        
        if (results_euclidean_training(i,1) < results_euclidean_training(i,2))
            % distance from average female is less than distance from
            % average male
            results_euclidean_training(i,3) = 1.0;
        end
        if (results_euclidean_training(i,4) < results_euclidean_training(i,5))
            % distance from average female is less than distance from
            % average male
            results_euclidean_training(i,6) = 1.0;
        end
    else
        % the test example is male
        
        if (results_euclidean_training(i,2) <= results_euclidean_training(i,1))
            % distance from average male is less than or equal to
            % than distance from the average female
            results_euclidean_training(i,3) = 1.0;
        end
        if (results_euclidean_training(i,5) <= results_euclidean_training(i,4))
            % distance from average male is less than or equal to
            % than distance from the average female
            results_euclidean_training(i,6) = 1.0;
        end 
       
    end
    
end

% BASELINE TEST FINISHED *********************************

% *********************************************************
% Main test area, using PCA compression-recompression distance
% as a classifier.
% Have the training and test data loaded, now loop, using different
% number of eigenvectors

% the inner loops starts with 1 top eigenvector, adding 1
% eigenvector per iteration, stopping at a max of top_num_eigs
% since there are 342 male and 341 female training examples,
% stop at 341 max.

%top_num_eigs = num_female_training;
%set this number to one when doing figure work
num_top_eigs = 60;
times = zeros(1,num_top_eigs);

% create a 3d matrix of results for the eigen-projection classifier.
% row number (dim 1) corresponds to test example number
% column (dim 2) corresponds to test statistic gathered
% the third dimension represents the number of eigenvectors used.
% col 1 = residual distance from female eigenspace reprojection grayscale
% col 2 = residual distance from male eigenspace reprojection grayscale
% col 3 = grayscale score for this example, at this number of eigs (0.0 for
% incorrect, 1.0 for correct)
% col 4 = residual distance from female eigenspace reprojection sxs
% col 5 = residual distance from male eigenspace reprojection sxs
% col 6 = sxs score for this example, at this number of eigs
results_pca_test = zeros(num_female_test + num_male_test, 6, num_top_eigs);
results_pca_training = zeros(num_female_training + num_male_training, 6, num_top_eigs);

for i = 1:num_top_eigs
    fprintf('Starting classification loop with %d eigenvectors\n', i);
    tic;
    %create the sub_arrays of eigenvectors (for clarity in code)
    female_subset_eigs_gray = female_eigs_gray(:, num_female_training-i+1:num_female_training);
    female_subset_eigs_sxs = female_eigs_sxs(:, num_female_training-i+1:num_female_training);
    
    male_subset_eigs_gray = male_eigs_gray(:, num_male_training-i+1:num_male_training);
    male_subset_eigs_sxs = male_eigs_sxs(:, num_male_training-i+1:num_male_training);
    
    
    %     this code would project onto the span of all the eigs
    %     %project test data onto the span the eigenvectors for the different
    %     %tests
    %     Ttest_gray_female = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_female_gray)*female_eigs_gray;
    %     Ttest_gray_male = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_male_gray)*male_eigs_gray;
    %     Ttest_sxs_female = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_female_sxs)*female_eigs_sxs;
    %     Ttest_sxs_male = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_male_sxs)*male_eigs_sxs;
    %
    %     %project training data onto span
    %     Ttrain_gray_female = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_female_gray)*female_eigs_gray;
    %     Ttrain_gray_male = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_male_gray)*male_eigs_gray;
    %     Ttrain_sxs_female = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_female_sxs)*female_eigs_sxs;
    %     Ttrain_sxs_male = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_male_sxs)*male_eigs_sxs;
    
    %project test data onto the span the eigenvectors for the different
    %tests. Use the subsets of eigs selected in the loop
    Ttest_gray_female = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_female_gray)*female_subset_eigs_gray;
    Ttest_gray_male = (test_data_gray - ones(num_female_test+num_male_test,1)*avg_male_gray)*male_subset_eigs_gray;
    Ttest_sxs_female = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_female_sxs)*female_subset_eigs_sxs;
    Ttest_sxs_male = (test_data_sxs - ones(num_female_test+num_male_test,1)*avg_male_sxs)*male_subset_eigs_sxs;
    
    Ttrain_gray_female = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_female_gray)*female_subset_eigs_gray;
    Ttrain_gray_male = (training_data_gray - ones(num_female_training+num_male_training,1)*avg_male_gray)*male_subset_eigs_gray;
    Ttrain_sxs_female = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_female_sxs)*female_subset_eigs_sxs;
    Ttrain_sxs_male = (training_data_sxs - ones(num_female_training+num_male_training,1)*avg_male_sxs)*male_subset_eigs_sxs;
    
    %classify all test data
    for j = 1:num_female_test+num_male_test
        %these weight vectors are the coordinates 
        %in the new basis of eigenvectors
        w_gray_female = Ttest_gray_female(j,:);
        w_gray_male = Ttest_gray_male(j,:);
        
        w_sxs_female = Ttest_sxs_female(j,:);
        w_sxs_male = Ttest_sxs_male(j,:);
        
        reproject_f_gray = w_gray_female*female_subset_eigs_gray' + avg_female_gray;
        reproject_m_gray = w_gray_male*male_subset_eigs_gray' + avg_male_gray;
        
        reproject_f_sxs = w_sxs_female*female_subset_eigs_sxs' + avg_female_sxs;
        reproject_m_sxs = w_sxs_male*male_subset_eigs_sxs' + avg_male_sxs;
        
        % test the test data euclidean distance
        
        results_pca_test(j,1,i) = norm(test_data_gray(j,:) - reproject_f_gray);
        results_pca_test(j,2,i) = norm(test_data_gray(j,:) - reproject_m_gray);
        
        results_pca_test(j,4,i) = norm(test_data_sxs(j,:) - reproject_f_sxs);
        results_pca_test(j,5,i) = norm(test_data_sxs(j,:) - reproject_m_sxs);
        
        if (j < 39)
            % the test example is female
            
            if (results_pca_test(j,1,i) < results_pca_test(j,2,i))
                % distance from average female is less than distance from
                % average male
                results_pca_test(j,3,i) = 1.0;
            end
            if (results_pca_test(j,4,i) < results_pca_test(j,5,i))
                % distance from average female is less than distance from
                % average male
                results_pca_test(j,6,i) = 1.0;
            end
        else
            % the test example is male
            
            if (results_pca_test(j,2,i) <= results_pca_test(j,1,i))
                % distance from average male is less than or equal to
                % than distance from the average female
                results_pca_test(j,3,i) = 1.0;
            end
            if (results_pca_test(j,5,i) <= results_pca_test(j,4,i))
                % distance from average male is less than or equal to
                % than distance from the average female
                results_pca_test(j,6,i) = 1.0;
            end
            
        end
        
    end
    
    %classify all training data
    for j = 1:num_female_training+num_male_training
        %these weight vectors are the coordinates 
        %in the new basis of eigenvectors
        w_gray_female = Ttrain_gray_female(j,:);
        w_gray_male = Ttrain_gray_male(j,:);
        
        w_sxs_female = Ttrain_sxs_female(j,:);
        w_sxs_male = Ttrain_sxs_male(j,:);
        
        reproject_f_gray = w_gray_female*female_subset_eigs_gray' + avg_female_gray;
        reproject_m_gray = w_gray_male*male_subset_eigs_gray' + avg_male_gray;
        
        reproject_f_sxs = w_sxs_female*female_subset_eigs_sxs' + avg_female_sxs;
        reproject_m_sxs = w_sxs_male*male_subset_eigs_sxs' + avg_male_sxs;
        
        % test the test data euclidean distance
        
        results_pca_training(j,1,i) = norm(training_data_gray(j,:) - reproject_f_gray);
        results_pca_training(j,2,i) = norm(training_data_gray(j,:) - reproject_m_gray);
        
        results_pca_training(j,4,i) = norm(training_data_sxs(j,:) - reproject_f_sxs);
        results_pca_training(j,5,i) = norm(training_data_sxs(j,:) - reproject_m_sxs);
        
        if (j <= num_female_training)
            % the test example is female
            
            if (results_pca_training(j,1,i) < results_pca_training(j,2,i))
                % distance from average female is less than distance from
                % average male
                results_pca_training(j,3,i) = 1.0;
            end
            if (results_pca_training(j,4,i) < results_pca_training(j,5,i))
                % distance from average female is less than distance from
                % average male
                results_pca_training(j,6,i) = 1.0;
            end
        else
            % the test example is male
            
            if (results_pca_training(j,2,i) <= results_pca_training(j,1,i))
                % distance from average male is less than or equal to
                % than distance from the average female
                results_pca_training(j,3,i) = 1.0;
            end
            if (results_pca_training(j,5,i) <= results_pca_training(j,4,i))
                % distance from average male is less than or equal to
                % than distance from the average female
                results_pca_training(j,6,i) = 1.0;
            end
            
        end
        
    end
    time_elapsed = toc;
    fprintf('Ending classification loop with %d eigenvectors, time elapsed (s) :%d\n', i, time_elapsed);
    times(j)=time_elapsed;
    
end

% scratch
% w = Ttest_gray_female(1,:);
% fproj_w = w*female_subset_eigs_gray';
% imagesc(reshape(fproj_w+avg_female_gray, nrow, ncol));
