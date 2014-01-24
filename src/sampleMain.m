loadAnnotations();

[dictionary, sigma_inv_half, mu] = buildDictionary();

cls = 'car';

[model] = trainSVM(cls, dictionary, sigma_inv_half, mu);

evaluateModel(model, cls, sigma_inv_half, mu); % Evaluate model with val

%TODO: We need to predict the test set here

