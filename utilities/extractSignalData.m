function [signalData, annotationsData, signalIds, sampFrequencies] = extractSignalData(loadedSignals)
    % Input:
    %   - loadedSignals: Structure containing loaded signals and annotations
    % Outputs:
    %   - signalData: Cell
    %   - annotationsData: Cell
    %   - signalIds: Cell
    %   - sampFrequencies: Cell
    % Extract field names for signals
    signalFieldNames = fieldnames(loadedSignals);
    signalFieldNames = signalFieldNames(strncmp(signalFieldNames, 'sig_', 4));

    % Extract data
    signalData = cellfun(@(x) x.data, struct2cell(loadedSignals), 'UniformOutput', false);
    annotationsData = cellfun(@(x) x.artif, struct2cell(loadedSignals), 'UniformOutput', false);
    sampFrequencies = cellfun(@(x) x.samplingFreq, struct2cell(loadedSignals), 'UniformOutput', false);
    signalIds = signalFieldNames; % Store signal IDs (field names)
end