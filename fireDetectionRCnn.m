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

% Combine datastore
cds = combine(imageDS, boxLabelDS);

% Creating CNN Object Detection Network
inputSize = [227 227 3];

% Defining object classes
classNames = {'Fire'};

% Number of object classes to detect
numClasses = width(Dataset)-1;

% CNN layer
lgraph = createRCNN(inputSize, numClasses, 'alexnet');

% Augment the training data
augmentedTrainingData = transform(cds,@augmentData);

% Preprocess the augmented training data to prepare for training
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

% Custom CNN Object Detector Hyperparameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ....
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.8, ...
    'MaxEpochs', 10, ...
    'VerboseFrequency', 50, ...
    'CheckpointPath', tempdir, ...
    'Shuffle','every-epoch');

if doTraining
    % Train the CNN detector with the data.
    [detector, info] = trainRCNNObjectDetector(trainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    detector = pretrained.detector
end

% Displaying Example
gpuDevice(1);
I = imread('100.jpg');
I = imresize(I,inputSize(1:2));
[fire, scores] = detect(detector, I);
threshold = 0.3;
idx = scores > threshold;
filtered_fire = fire(idx, :);
filtered_scores = scores(idx);

I = insertObjectAnnotation(I, 'rectangle', filtered_fire, filtered_scores);
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
