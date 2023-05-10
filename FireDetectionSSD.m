clc; clear;
doTraining = true;

% Loading the custom dataset
myDataSet = load('fireDataset.mat');
imageSource = myDataSet.gTruth.DataSource.Source;
fire =  myDataSet.gTruth.LabelData.fire;
Dataset = table(imageSource, fire);

% Display first 6 rows personDataset.
Dataset(1:6,:)

% Shuffle training data
rng(0);
shuffledIdx = randperm(height(Dataset));
trainingData = Dataset(shuffledIdx,:);

% Image and label datastore
imageDS = imageDatastore(trainingData.imageSource);
boxLabelDS = boxLabelDatastore(trainingData(:,2:end));

%Combining datastore
cds = combine(imageDS, boxLabelDS);

%Creating SSD Object Detection Network
inputSize = [300 300 3];

% Defining object classes
classNames = {'Fire'};

% Number of object classes to detect
numClasses = width(Dataset)-1;

% SSD layer
lgraph = ssdLayers(inputSize, numClasses, 'resnet50');

% Augment the training data
augmentedTrainingData = transform(cds,@augmentData);

% Preprocess the augmented training data to prepare for training
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

% Custom SSD Object Detector Hyperparameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ....
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 50, ...
    'VerboseFrequency', 50, ...
    'CheckpointPath', tempdir, ...
    'Shuffle','every-epoch');



if doTraining
    % Train the SSD detector with the data.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
     detector = pretrained.detector
end

% Displaying Example
I = imread('100.jpg');
I = imresize(I,inputSize(1:2));
[fire,scores] = detect(detector,I, Threshold=0.2);
I = insertObjectAnnotation(I,'rectangle',fire,scores);
figure
imshow(I)

% Load test data
myData2 = load('fireTestSet.mat');
imageSource = myData2.gTruth.DataSource.Source(:,1);
fire =  myData2.gTruth.LabelData.fire;
Dataset2 = table(imageSource, fire);
% Image datastore
imds2 = imageDatastore(imageSource);
% label datastore
blds2 = boxLabelDatastore(Dataset2(:,'fire'));
% combine
testDs = combine(imds2,blds2);
detectionResults = detect(detector, imds2,Threshold=0.1);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, blds2);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap)) 