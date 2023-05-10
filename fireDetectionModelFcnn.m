clear; clc;
doTraining = true;
% Load data
myDataSet = load('fireDataset.mat');
imageSource = myDataSet.gTruth.DataSource.Source(:,1);
fire =  myDataSet.gTruth.LabelData.fire;
Dataset = table(imageSource, fire);
% Shuffling train data
gpuDevice(1)
rng(0);
shuffledIndex = randperm(height(Dataset));
trainingDataset = Dataset(shuffledIndex,:);
% Image datastore
imageDs = imageDatastore(trainingDataset.imageSource);
% label datastore
boxLabelds = boxLabelDatastore(trainingDataset(:,2:end));
% combine
ds = combine(imageDs,boxLabelds);
inputSize = [224 224 3];
numClasses = width(Dataset)-1;
trainingDataEstimation = transform(ds,@(data)preprocessData(data,inputSize));
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataEstimation, 7);
featureExtractionNetwork = mobilenetv2;
featureLayer = 'out_relu';
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
augmentedTrainingData = transform(ds,@augmentData);
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
gpuDevice(1)
options = trainingOptions('adam', ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',50);
if doTraining       
    % Train the Faster R-CNN detector.
    [detector,info] = trainFasterRCNNObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    detector = fasterRCNNObjectDetector('resnet50-coco'); 
end

I = imread('100.jpg');
I = imresize(I,inputSize(1:2));
[fire,scores] = detect(detector,I, Threshold=0.1);
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
imds2 = transform(imds2, @(x) imresize(x, inputSize(1:2))); % Resize test images
% label datastore
blds2 = boxLabelDatastore(Dataset2(:,'fire'));
% combine
testDs = combine(imds2,blds2);
detectionResults = detect(detector, imds2, 'Threshold', 0.1, 'MiniBatchSize', 1);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, blds2);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))



