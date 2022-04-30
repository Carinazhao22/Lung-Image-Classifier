%% SFU CMPT 340 Project Group 7 - Lung Virus Classifier, Apr 2020
%  by Sabrina Dalen, Jin Lu, Hui Wu, Caijie Zhao, Huiyi Zou
%  
%  Objective: To classify lung X-ray images as normal, COVID-19, 
%  or viral pneumonia using CNN, KNN, and RF
%
%  Instructions:
%  Install Deep Learning Toolbox for Neural Network cnn.
%  Install Computer Vision and Image Processing Toolbox for knn and rf.
%
%  Run files in order:
%  1. create_cnn.m - output trained network myCNN.mat using MATLAB
%       function trainNetwork 
%  2. cnn_classifier.m - classify test data using cnn trained network
%       myNet and output myCNNresults.mat
%  3. knn_rf_classifier.m - classify test data using knn and rf
%       MATLAB functions and output features.mat
%  4. feature_analysis.m - plot charts for analysis
%
%%

clear; % clear vars

% Load CNN trained network
load ('myCNN.mat');
fprintf('network loaded\n')

% Load test images into imds datastore collection of image files
testDataPath = fullfile(pwd,'Data','test');
test_imds = imageDatastore(testDataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% Create table of number of test images in each group (folder)
labelCount = countEachLabel(test_imds);
fprintf('dataset with %d images loaded\n',length(test_imds.Files));

%%
Normal.actual = double(labelCount{2,2});
Normal.true_covid = 0;
Normal.true_pneumonia = 0;
Normal.correct = 0;
Normal.count = 0;

Covid19.actual = double(labelCount{1,2});
Covid19.true_normal = 0;
Covid19.true_pneumonia = 0;
Covid19.correct = 0;
Covid19.count = 0;

Pneumonia.actual = double(labelCount{3,2});
Pneumonia.true_normal = 0;
Pneumonia.true_covid = 0;
Pneumonia.correct = 0;
Pneumonia.count = 0;

CNN_normal=[];
CNN_covid19=[];
CNN_pneumonia=[];
%%

for i = 1:length(test_imds.Files)
    if mod(i,100) == 0
        fprintf('%d images classified, %d remaining\n',i,length(test_imds.Files)-i);
    end
    
    % Resize images if necessary to 227 x 227
    x_ray_img = readimage(test_imds,i);
    if size(x_ray_img,3) ~= 3
        x_ray_img = x_ray_img(:,:,[1 1 1]);
    end
    resized_img = imresize(x_ray_img,[227 227]);
    
    % Classify test image using trained network myNet where
    % pred = predicted class label, score = predicted responses
    [pred,score] = classify(myNet,resized_img);
    diagnosis = sprintf(char(pred));
    img_label = sprintf(char(test_imds.Labels(i)));

    % Count accuracy of results per class label
    if diagnosis == "COVID-19" % Type 1
        Covid19.count = Covid19.count + 1;
        CNN_covid19=[CNN_covid19;test_imds.Files(i),1];
        if img_label == "COVID-19"
            Covid19.correct = Covid19.correct+1;
        elseif img_label == "NORMAL"
            Covid19.true_normal = Covid19.true_normal+1;
        elseif img_label == "PNEUMONIA"
            Covid19.true_pneumonia = Covid19.true_pneumonia + 1;
        else
            fprintf('empty label\n');
            break;
        end
    elseif diagnosis == "NORMAL" % Type 2
        Normal.count = Normal.count + 1;
        CNN_normal=[CNN_normal;test_imds.Files(i),2];
        if img_label == "COVID-19"
            Normal.true_covid = Normal.true_covid+1;
        elseif img_label == "NORMAL"
            Normal.correct = Normal.correct + 1;
        elseif img_label == "PNEUMONIA"
            Normal.true_pneumonia = Normal.true_pneumonia + 1;
        else
            fprintf('empty label\n');
            break;
        end
    elseif diagnosis == "PNEUMONIA" % Type 3
        Pneumonia.count = Pneumonia.count + 1;
        CNN_pneumonia=[CNN_pneumonia;test_imds.Files(i),3];
        if img_label == "COVID-19"
            Pneumonia.true_covid = Pneumonia.true_covid+1;
        elseif img_label == "NORMAL"
            Pneumonia.true_normal = Pneumonia.true_normal + 1;
        elseif img_label == "PNEUMONIA"
            Pneumonia.correct = Pneumonia.correct + 1;
        else
            fprintf('empty label\n');
            break;
        end
    else
        fprintf('empty diagnosis\n');
        break;
    end
        
end

CNN_classify=[CNN_normal; CNN_covid19; CNN_pneumonia];
save CNNresults CNN_classify;
fprintf('Total %d images classified\n',length(test_imds.Files));

%%
CovidTPR = Covid19.correct / Covid19.count;
NormalTPR = Normal.correct / Normal.count;
PneumoniaTPR = Pneumonia.correct / Pneumonia.count;
fprintf('Covid TPR: %f\n',CovidTPR);
fprintf('Normal TPR: %f\n',NormalTPR);
fprintf('Pneumonia TPR: %f\n',PneumoniaTPR);

%%
Labels = {'Normal';'COVID-19';'Pneumonia';'Total';'True Rate'};
CNN_Classified_Normal = [Normal.correct;Normal.true_covid;Normal.true_pneumonia;Normal.count;NormalTPR];
CNN_Classified_COVID19 = [Covid19.true_normal;Covid19.correct;Covid19.true_pneumonia;Covid19.count;CovidTPR];
CNN_Classified_Pneumonia = [Pneumonia.true_normal;Pneumonia.true_covid;Pneumonia.correct;Pneumonia.count;PneumoniaTPR];
Total_Actual = [Normal.actual;Covid19.actual;Pneumonia.actual;Normal.actual+Covid19.actual+Pneumonia.actual;NaN];

% TrueRate = [NormalTPR;CovidTPR;PneumoniaTPR;NaN];
T = table(Labels,CNN_Classified_Normal,CNN_Classified_COVID19,CNN_Classified_Pneumonia,Total_Actual);
fig = uifigure('Position',[500 500 750 350]);
uit = uitable(fig,'Data',T,'Position',[20 20 710 200]);
color_row = [1;2;3];
color_col = [2;3;4];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);

