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

% Load features file
load('features.mat');
features = zero_degree_features;
covid=[];
normal=[];
pneumonia=[];
for i=1:size(features,1)
    if features(i,7)==1
        covid = [covid;features(i,:)];
    elseif features(i,7)==2
        normal = [normal;features(i,:)];
    else
        pneumonia = [pneumonia;features(i,:)];
    end
end

%% contrast scatter plot

figure(1);
subplot(2,3,1);
plot(normal(:,1),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,1),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,1),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('Contrast','FontWeight','bold');

%% correlation scatter plot
subplot(2,3,2);
plot(normal(:,2),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,2),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,2),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('Correlation','FontWeight','bold');


%% energy scatter plot
subplot(2,3,3);
plot(normal(:,3),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,3),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,3),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('Energy','FontWeight','bold');


%% homogeneity scatter plot
subplot(2,3,4);
plot(normal(:,4),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,4),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,4),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('Homogeneity','FontWeight','bold');


%% entropy scatter plot
subplot(2,3,5);
plot(normal(:,5),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,5),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,5),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('Entropy','FontWeight','bold');


%% IDE scatter plot
subplot(2,3,6);
plot(normal(:,6),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,6),2*ones(1,size(covid,1)),'o');
plot(pneumonia(:,6),1*ones(1,size(pneumonia,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Pneumonia','COVID-19','Normal'});
title('IDE','FontWeight','bold');


sgtitle('Scatter Plots of Texture Features','FontWeight','bold');