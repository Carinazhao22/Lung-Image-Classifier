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
clc; % clear command window

% Load training images into imds datastore, a collection of image files
trainDataPath = fullfile(pwd,'Data','train');
imds = imageDatastore(trainDataPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% Create table showing number of training images in each group
labelCount = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');

%%
AlexNetLayers = [
    %1
    imageInputLayer([227 227 3])
    %2
    convolution2dLayer([3 3],96,'Stride',4)
    %3
    reluLayer
    %4
    crossChannelNormalizationLayer(5)
    %5
    maxPooling2dLayer([3 3],'Stride',[2 2])
    %6
    groupedConvolution2dLayer([5 5],128,2,'Stride',1,'Padding',2)
    %7
    reluLayer
    %8
    crossChannelNormalizationLayer(5)
    %9
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding',0)
    %10
    convolution2dLayer([3 3],384,'Stride',1,'Padding',1)
    %11
    reluLayer
    %12
    groupedConvolution2dLayer([3 3],192,2,'Stride',1,'Padding',1)
    %13
    reluLayer
    %14
    groupedConvolution2dLayer([3 3],192,2,'Stride',1,'Padding',1)
    %15
    reluLayer
    %16
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding',0)
    %17
    fullyConnectedLayer(1024)
    %18
    reluLayer
    %19
    dropoutLayer(0.5)
    %20
    fullyConnectedLayer(1024)
    %21
    reluLayer
    %22
    dropoutLayer(0.5)
    %23
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
%%

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Use MATLAB trainNetwork to train new network to classify images:
%   imds - stores input image data
%   layers - network architecture
%   options - training options
% https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html
myNet = trainNetwork(imdsTrain,AlexNetLayers,options);
YPred = classify(myNet,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
save ('myCNN.mat');