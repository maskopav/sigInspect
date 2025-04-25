function [loadedSignals] = loadSignalsFromMat(matFilePath, saveData, savePath)
% Loads signal data and annotations from a .mat file into structure.
%
% Inputs:
% - matFilePath: Full path to the .mat file containing data
% - saveData (optional): Boolean indicating whether to save the data (default: false)
% - savePath (optional): Path to save the .mat file (default: 'loadedSignals.mat' in current directory)
%
% Outputs:
% - loadedSignals: Structure containing loaded signals and annotations

if nargin < 2
    saveData = false; % Default: do not save data
end

[parentDir, ~, ~] = fileparts(matFilePath);
if nargin < 3
    [grandParentDir, ~, ~] = fileparts(parentDir);
    savePath = fullfile(grandParentDir, 'loadedSignalsCZSK.mat'); % Default save path
end

if saveData
    fprintf(['Loading signals with annotations has started, ' ...
        'output will be saved into: %s\n'], savePath);
else
    fprintf(['Loading signals with annotations has started, ' ...
        'output will not be saved\n']);
end

% Load the .mat file containing the struct infoCZSK
fprintf('Loading data from: %s\n', matFilePath);
matData = load(matFilePath);

% Check if the infoCZSK field exists in the loaded data
if ~isfield(matData, 'infoCZSK')
    error('The mat file does not contain the infoCZSK structure.');
end

infoCZSK = matData.infoCZSK;

% Initialize a structure to store loaded signals
loadedSignals = struct();

% Debug information about the structure
try
    % Determine the number of signals from a field that exists
    numSignals = numel(infoCZSK);
    
    fprintf('Total number of signals detected: %d\n', numSignals);
    
    % Loop through signals with annotations
    progressMarks = round([0.25, 0.50, 0.75, 1] * numSignals);
    annotationCounter = 0;
    
    for i = 1:numSignals
        metadataStruct = infoCZSK(i);
        annotationCounter = annotationCounter + 1;
        
        % Get the signal ID
        sigId = metadataStruct.sigId;

        signalPath = fullfile(parentDir, metadataStruct.path);
        fprintf('Processing signal %d/%d: %s at %s\n', annotationCounter, numSignals, sigId, signalPath);
        
        % Load the signal data
        if exist(signalPath, 'file')
            signalData = load(signalPath);
            
            % Get field names from the loaded signal data
            dataFieldNames = fieldnames(signalData);
            if ~isempty(dataFieldNames)
                % Assuming the signal data is stored in the first field
                signalMatrix = signalData.(dataFieldNames{1});
                artifactsMatrix = metadataStruct.artifacts;

                % Store channels as separate signals
                for channel=1:size(signalMatrix,1)
                    channelFieldName = matlab.lang.makeValidName(['sig_', sigId, '_ch', num2str(channel)]);

                    % Store the signal and annotations in the structure
                    loadedSignals.(channelFieldName).data = signalMatrix(channel, :); % Channel data
                    loadedSignals.(channelFieldName).sigId = sigId; % Channel ID
                    loadedSignals.(channelFieldName).channelNumber = channel; % Store channel number
                    loadedSignals.(channelFieldName).artif = artifactsMatrix(channel, :); 
                    
                    % Add all available fields from infoCZSK
                    fields = {'samplingFreq', 'Nsamples', 'Nchannels', 'info', 'center', 'artifactAuthors', 'patient'};
                    for f = 1:numel(fields)
                        fieldName = fields{f};
                        if isfield(metadataStruct, fieldName)
                            fieldValue = metadataStruct.(fieldName); % Get the value
                             % if iscell(fieldValue)
                             %     fieldValue = fieldValue{1};
                             % end
                            loadedSignals.(channelFieldName).(fieldName) = fieldValue;
                        end
                    end
                end
            else
                warning('Signal file %s has no fields.', signalPath);
            end
        else
            warning('Signal file %s does not exist.', signalPath);
        end
        
        % Report progress at 25%, 50%, and 75%
        if ismember(annotationCounter, progressMarks)
            fprintf('Progress: %.0f%% signals loaded\n', (annotationCounter / numSignals) * 100);
        end
    end
    
    % Optionally save the data
    if saveData
        disp('Saving the data..')
        save(savePath, 'loadedSignals', '-v7.3');
        fprintf('Data saved to %s\n', savePath);
    end
    
    disp('Signals with annotations successfully loaded.');
catch e
    fprintf('Error occurred: %s\n', e.message);
    disp('Stack trace:');
    disp(e.stack);
    rethrow(e);
end
end