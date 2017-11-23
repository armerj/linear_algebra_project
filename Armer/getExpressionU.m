function U = getExpressionU(path, pictureType)
    % get U of the training data
    B = zeros(4096, 16); %intialize

    cd(path) %change current dir

    files = dir(pictureType); % get all files ending in pictureType

    % for each file in files
    divisor = 255.0*ones(64*64, 1);
    for i = 1:length(files) % loop through files
        A = imread(files(i).name); 
        A = imresize(A, [64 64]); %shrink the image
        A = double((reshape(A,64*64, 1)))./divisor; % reshape to 1 column
        B(:, i) = A; % build matrix of column vectors
    end

    %B = U*S*V'
    [U,S,V] = svd(B,0); % get SVD
end