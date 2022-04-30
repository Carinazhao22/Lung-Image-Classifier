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

%% -------Machine learning method----------- %%
%% Load training data 
trainDataPath=imageSet('Data\train','recursive');
trainData=partition(trainDataPath,107,'randomize'); %107 samples of each type
Covid19=trainData(1);
Normal=trainData(2);
Pneumonia=trainData(3);

% Initialize model
trainResult = [];
trainType = [];

% COVID-19, Type 1
for i=1:Covid19.Count
    img = imread(string(Covid19.ImageLocation(i)));
    trainResult = [trainResult;processGLCM(img)];
    trainType = [trainType;1];
end
 
% Normal, Type 2
for i=1:Normal.Count
    img = imread(string(Normal.ImageLocation(i)));
    trainResult = [trainResult;processGLCM(img)];
    trainType = [trainType;2];
end 
 
% Pneumonia, Type 3
for i=1:Pneumonia.Count
    img = imread(string(Pneumonia.ImageLocation(i)));
    trainResult = [trainResult;processGLCM(img)];
    trainType = [trainType;3];
end

% Texture features at 0-degree direction stored in columns 1, 5, 9, etc.
zero_degree_features=[];
for i=1:4:24
	zero_degree_features = [zero_degree_features, trainResult(:,i)];
end
zero_degree_features = [zero_degree_features, trainType];
save features zero_degree_features;
 
%% Load test data
testData=imageSet('Data\test','recursive');
Covid19=testData(1);
Normal=testData(2);
Pneumonia=testData(3);
testResult=[];

% COVID-19, Type 1
for i=1:Covid19.Count
    img=imread(string(Covid19.ImageLocation(i)));
    testResult=[testResult;processGLCM(img),1];
end

% Normal, Type 2
for i=1:Normal.Count
    img=imread(string(Normal.ImageLocation(i)));
    testResult=[testResult;processGLCM(img),2];
end
 
% Pneumonia, Type 3
for i=1:Pneumonia.Count
    img=imread(string(Pneumonia.ImageLocation(i)));
    testResult=[testResult;processGLCM(img),3];
end

%% Random Forest
nTrees = 500;
B = TreeBagger(nTrees,trainResult,trainType, ...
               'OOBPrediction','On','Method','classification'); 
%view(B.Trees{1},'Mode','graph')
%view(B.Trees{2},'Mode','graph')
figure(2);
oobErrorBaggedEnsemble = oobError(B);
fig2 = plot(oobErrorBaggedEnsemble);
xlabel('Number of Grown Trees')
ylabel('Out-of-bag Classification Error')
rffit = predict(B,testResult(:,1:end-1)); 
rffit = cell2mat(rffit); 

%% KNN
% Find best hyperparameters in KNN (K & distance)
rng(1) % initialize random number generator using seed of 1
Mdl = fitcknn(trainResult,trainType,'OptimizeHyperparameters', ...
             'auto','HyperparameterOptimizationOptions', ...
             struct('AcquisitionFunctionName','expected-improvement-plus'));
K = fitcknn(trainResult,trainType,'NumNeighbors', ...
            Mdl.NumNeighbors,'Distance',Mdl.Distance,'Standardize',1);
knnfit = predict(K,testResult(:,1:end-1));
a = testResult(:,end); % test type
 
%% Confusion Matrix from knnR and rfR  
rfCorrCovid=0;
rfCorrNormal=0;
rfCorrVirus=0;
knnCorrCovid=0;
knnCorrNormal=0;
knnCorrVirus=0;
actualCovid=0;
actualNormal=0;
actualVirus=0;
knnErrAsCovid=0;
knnErrAsVirus=0;
rfErrAsCovid=0;
rfErrAsVirus=0;
 
% Normal
for i=1:size(a,1)
    if a(i)==2
        actualNormal=actualNormal+1;
        if knnfit(i)==2
            knnCorrNormal=knnCorrNormal+1;
        elseif knnfit(i)==1
            knnErrAsCovid=knnErrAsCovid+1;
        else
            knnErrAsVirus=knnErrAsVirus+1;
        end
        
        if rffit(i)==2
            rfCorrNormal=rfCorrNormal+1;
        elseif rffit(i)==1
            rfErrAsCovid=rfErrAsCovid+1;
        else
            rfErrAsVirus=rfErrAsVirus+1;
        end
    end             
end
knnR=[knnCorrNormal knnErrAsCovid knnErrAsVirus,actualNormal];
rfR=[rfCorrNormal rfErrAsCovid rfErrAsVirus,actualNormal];
 
% COVID-19
knnErrAsNormal=0;
knnErrAsVirus=0;
rfErrAsNormal=0;
rfErrAsVirus=0;
 
for i=1:size(a,1)
    if a(i)==1
       actualCovid=actualCovid+1;
        if knnfit(i)==1
            knnCorrCovid=knnCorrCovid+1;
        elseif knnfit(i)==2
            knnErrAsNormal=knnErrAsNormal+1;
        else
            knnErrAsVirus=knnErrAsVirus+1;
        end
        if rffit(i)==1
            rfCorrCovid=rfCorrCovid+1;
        elseif rffit(i)==2
            rfErrAsNormal=rfErrAsNormal+1;
        else
            rfErrAsVirus=rfErrAsVirus+1;
        end        
    end               
end
knnR=[knnR;knnErrAsNormal,knnCorrCovid,knnErrAsVirus,actualCovid];
rfR=[rfR;rfErrAsNormal,rfCorrCovid,rfErrAsVirus,actualCovid];
 
% Pneumonia
knnErrAsCovid=0;
knnErrAsNormal=0;
rfErrAsCovid=0;
rfErrAsNormal=0;

for i=1:size(a,1)
    if a(i)==3
        actualVirus=actualVirus+1;
        if knnfit(i)==3
            knnCorrVirus=knnCorrVirus+1;
        elseif knnfit(i)==1
            knnErrAsNormal=knnErrAsNormal+1;
        else
            knnErrAsCovid=knnErrAsCovid+1;
        end
        if rffit(i)==3
            rfCorrVirus=rfCorrVirus+1;
        elseif rffit(i)==1
            rfErrAsCovid=rfErrAsCovid+1;
        else
            rfErrAsCovid=rfErrAsCovid+1;
        end        
    end               
end
knnR=[knnR;knnErrAsNormal,knnErrAsCovid,knnCorrVirus,actualVirus];
rfR=[rfR;rfErrAsNormal,rfErrAsCovid,rfCorrVirus,actualVirus];

 
%% Display confusion matrix
totalSumK=[sum(knnR(:,1)),sum(knnR(:,2)),sum(knnR(:,3)),sum(knnR(:,4))];
totalSumR=[sum(rfR(:,1)),sum(rfR(:,2)),sum(rfR(:,3)),sum(rfR(:,4))];
TPRk=[knnR(1,1)/totalSumK(1),knnR(2,2)/totalSumK(2),knnR(3,3)/totalSumK(3),NaN];
TPRr=[rfR(1,1)/totalSumR(1),rfR(2,2)/totalSumR(2),rfR(3,3)/totalSumR(3),NaN];
knnR=[knnR;totalSumK;TPRk];
rfR=[rfR;totalSumR;TPRr];

CMknn=array2table(knnR,'VariableNames',{'KNN Classified Normal', ...
    'Classified COVID-19','Classified Pneumonia','Total Actual'}, ...
    'RowNames',{'Normal','COVID-19','Pneumonia','Total','True Rate'});
CMknn.Properties.Description = 'Confusion Matrix for KNN Results';

CMrf=array2table(rfR,'VariableNames',{'RF Classified Normal', ...
    'Classified COVID-19','Classified Pneumonia','Total Actual'}, ...
    'RowNames',{'Normal','COVID-19','Pneumonia','Total','True Rate'});
CMrf.Properties.Description = 'Confusion Matrix for RF Results';

figure(3);
fig = uifigure;
uit = uitable(fig,'Data',CMknn,'Position',[20 200 515 150]);
color_row = [1;2;3];
color_col = [1;2;3];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);
hold on;
uit = uitable(fig,'Data',CMrf,'Position',[20 200 515 150]);
hold off;
color_row = [1;2;3];
color_col = [1;2;3];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);

%% Display classification results showing this figure 
% only when using 5 test images for each set
lim = size(a,1)/3;
lim = ceil(1.5*lim);
acturalNo=0;
acturalCo=0;
acturalVi=0;
rfNo=0;
rfCo=0;
rfVi=0;
knnNo=0;
knnCo=0;
knnVi=0;
%figure;
flag=1;
flag1=1;
flag2=1;
flag3=1;
flag4=1;
flag5=1;
flag6=1;
flag7=1;
flag8=1;
%title('Classification Result');
for i=1:size(a,1)
    if i<=Covid19.Count
        file = Covid19.ImageLocation(i);
	elseif i>Covid19.Count && i<=Covid19.Count+Normal.Count
        file = Normal.ImageLocation(i-Covid19.Count);
    else
        file = Pneumonia.ImageLocation(i-Covid19.Count-Normal.Count);
    end
    
    if a(i)==1
        acturalCo=acturalCo+1;        
%         subplot(12,lim,lim+acturalCo);imshow(imread(string(file)));
%         if flag==1
%             title('Real situation COVID-19');
%             flag=0;
%         end
        
	elseif a(i)==2
        acturalNo=acturalNo+1;
%         subplot(12,lim,acturalNo);imshow(imread(string(file)));
%         if flag1==1
%             title('Real situation Normal');
%             flag1=0;
%         end

    else
        acturalVi=acturalVi+1;
%         subplot(12,lim,2*lim+acturalVi);imshow(imread(string(file)));
%         if flag2==1
%         title('Real situation Virus');
%         flag2=0;
%         end
    end
    
    if rffit(i)=='1'
        rfCo=rfCo+1;
%         subplot(12,lim,4*lim+rfCo);imshow(imread(string(file)));
%         if flag3==1
%             title('RF Covid-19');
%             flag3=0;
%         end
	elseif rffit(i)=='2'
        rfNo=rfNo+1;
%         subplot(12,lim,3*lim+rfNo);imshow(imread(string(file)));
%         if flag4==1
%             title('RF Normal');
%             flag4=0;
%         end
    else
        rfVi=rfVi+1;
%         subplot(12,lim,5*lim+rfVi);imshow(imread(string(file)));
%         if flag5==1
%             title('RF Virus');
%             flag5=0;
%         end
    end
    
    if knnfit(i)==1
        knnCo=knnCo+1;
%         subplot(12,lim,7*lim+knnCo);imshow(imread(string(file)));
%         if flag6==1
%         title('KNN Covid-19');
%         flag6=0;
%         end
	elseif knnfit(i)==2
        knnNo=knnNo+1;
%         subplot(12,lim,6*lim+knnNo);imshow(imread(string(file)));
%         if flag7==1
%         title('KNN Normal');
%         flag7=0;
%         end
    else
        knnVi=knnVi+1;
%         subplot(12,lim,8*lim+knnVi);imshow(imread(string(file)));
%         if flag8==1
%         title('KNN Virus');
%         flag8=0;
%         end
    end
end
% sgt = sgtitle('Real and Classified Image Results ','Color','Black');
% sgt.FontSize = 20;


%% show bar figure
figure(4);
Y = [acturalNo, rfNo, knnNo; acturalCo, rfCo, knnCo; acturalVi, rfVi, knnVi];
b = bar(Y);
ylim([0 lim]);
for i=1:3
text(b(i).XEndPoints,b(i).YEndPoints,string(b(i).YData), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom');
end
set(gca,'xticklabel',{'Normal','Covid-19','Virus'})
title('Number of Disease Classification Results via Different Algorithms','FontWeight','bold')
xlabel('Diseases','FontWeight','bold')
ylabel('Number of Each Type','FontWeight','bold')
legend('Actual','RF','KNN','FontWeight','bold')

%% -------Helper Methods at End of File----------- %%

% GLCM process to extract features
function [features] = processGLCM(img)
    if size(img,3)>1
        img = rgb2gray(img);
    end
    img = uint8(255*mat2gray(img));
    
    % Gray-level co-occurrence matrix (GLCM) with 4 directions (0,45,90,135)
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});

    % Normalization
    for n = 1:4
        glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
    end
    
    entropy = zeros(1,4);
    IDE = zeros(1,4);

    for n = 1:4
        for f = 1:8
            for j = 1:8
                if glcm(f,j,n)~=0
                    entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n);
                    % Inverse Differential Moment
                    IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2)); 
                end
            end
        end
    end
    features=[feature.Contrast, feature.Correlation, ...
              feature.Energy, feature.Homogeneity, entropy, IDE];
end