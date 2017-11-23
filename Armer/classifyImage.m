function [value, expression] = classifyImage(image, angerU, disgustU, surpiseU, sadU, happyU, fearU, neutralU)
% this function will get the residual value for each type of cassification. The smaller the residual value
% the more likely the face will fit that classification

    I = eye(4096);

    % residual = ||(I-U*Ut)*image||/norm(image), minimize this value
    angerValue = norm((I-angerU*transpose(angerU))*image)/norm(image); 
    disgustValue = norm((I-disgustU*transpose(disgustU))*image)/norm(image); 
    surpiseValue = norm((I-surpiseU*transpose(surpiseU))*image)/norm(image); 
    sadValue = norm((I-sadU*transpose(sadU))*image)/norm(image); 
    happyValue = norm((I-happyU*transpose(happyU))*image)/norm(image); 
    fearValue = norm((I-fearU*transpose(fearU))*image)/norm(image); 
    neutralValue = norm((I-neutralU*transpose(neutralU))*image)/norm(image);

    disp([angerValue, disgustValue, surpiseValue, sadValue, happyValue, fearValue, neutralValue])

    % get the mininal residual value
    [value, index] =  min([angerValue, disgustValue, surpiseValue, sadValue, happyValue, fearValue, neutralValue]);

    % determine which classification was determined
    switch index % determine what expression it was based on index
        case 1;
            expression = 'anger';
        case 2;
            expression = 'disgust';
        case 3;
            expression = 'surpise';
        case 4;
            expression = 'sad';
        case 5;
            expression = 'happy';
        case 6; 
            expression = 'fear';
        case 7; 
            expression = 'neutral';
    end;
end
