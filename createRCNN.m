function lgraph = createRCNN(inputSize, numClasses, baseNetwork)
    % Load the pretrained network
    net = alexnet; % You can replace this with another network, e.g., vgg16, resnet50, etc.
    
    % Extract the layer graph from the pretrained network
    lgraph = layerGraph(net);

    % Remove the last three layers of the pretrained network
    lgraph = removeLayers(lgraph, {'fc8', 'prob', 'output'});

    % Add new fully connected layer
    newFC = fullyConnectedLayer(numClasses + 1, 'Name', 'new_fc');
    lgraph = addLayers(lgraph, newFC);

    % Add new softmax layer
    newSoftmax = softmaxLayer('Name', 'new_softmax');
    lgraph = addLayers(lgraph, newSoftmax);

    % Add new classification layer
    newClassOutput = classificationLayer('Name', 'new_classoutput');
    lgraph = addLayers(lgraph, newClassOutput);

    % Connect the new layers to the pretrained network
    lgraph = connectLayers(lgraph, 'drop7', 'new_fc');
    lgraph = connectLayers(lgraph, 'new_fc', 'new_softmax');
    lgraph = connectLayers(lgraph, 'new_softmax', 'new_classoutput');
end
