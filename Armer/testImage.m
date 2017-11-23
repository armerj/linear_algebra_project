function result = testImage(trainingPath, TestPath, pictureType)
path = trainingPath;

%training data
%***********************************
angerU = getExpressionU(strcat(path,'\anger'), pictureType);
disgustU = getExpressionU(strcat(path,'\disgust'), pictureType);
surpiseU = getExpressionU(strcat(path,'\surpise'), pictureType);
sadU = getExpressionU(strcat(path,'\sad'), pictureType);
happyU = getExpressionU(strcat(path,'\happy'), pictureType);
fearU = getExpressionU(strcat(path,'\fear'), pictureType);
neutralU = getExpressionU(strcat(path,'\neutral'), pictureType);
%***********************************

% folders for test data
folders = {'anger' 'disgust' 'surpise' 'sad' 'happy' 'fear' 'neutral'};

for j = 1:length(folders) % loop through expressions
    numCorrect = 0;
    total = 0;
    path = strcat(TestPath, folders{j}, '\');
    disp(folders{j})
    cd(path) % chage dir 
    files = dir(pictureType); % get all files of pictureType

    for i = 1:length(files) % loop through files
        image = imread(files(i).name);  
        image = imresize(image, [64 64]); %shrink the image
        divisor = 255.0*ones(64*64, 1);
        image = double((reshape(image,64*64, 1)))./divisor; %reshape image
        
        %classify different images  
        [value, expression] = classifyImage(image, angerU, disgustU, surpiseU, sadU, happyU, fearU, neutralU);
        if (strcmp(expression, folders{j})) % add to count if correct
            numCorrect = numCorrect + 1;
        end
        total = total + 1; % running total of images
        
        % used to print data to file
        %fid = fopen('Results.txt','w');
        %fprintf(fid, strcat(files(i).name, ', ', value, ', ', expression, '\n'));
        %fclose(fid);
        
        % used to display data about file and expression
        %disp(strcat(files(i).name, ', ', value, ', ', expression));
    end

    disp(numCorrect/total); % display accuracy
end

end


        