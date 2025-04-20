ROOT_DIRECTORY = "data";  % Where every dataset is stored.

DATASET_NAME = "SEED";

EEG_FEATURE_TYPE = "de";  % Differential entropy feature.
EEG_SMOOTHING_METHOD = "LDS";  % Linear dynamic system smoothing.

INPUT_DIRECTORY = fullfile( ...
    ROOT_DIRECTORY, DATASET_NAME, "ExtractedFeatures");
OUTPUT_DIRECTORY = fullfile( ...
    ROOT_DIRECTORY, DATASET_NAME, EEG_FEATURE_TYPE + "_feature");

N_SUBJECTS = 15;
N_SESSIONS = 3;  % Per subject.
N_TRIALS = 15;  % Per session, per subject.

labelOfTrial = load(fullfile(INPUT_DIRECTORY, "label.mat")).label;

for iSubject = 1: N_SUBJECTS
    % Intent to collect "1_[SomeDateTime].mat"-like files in the directory.
    matFilesInfo = dir(fullfile(INPUT_DIRECTORY, iSubject + "_*.mat"));
    for iSession = 1: N_SESSIONS
        loadFileName = matFilesInfo(iSession).name;
        loadFilePath = fullfile(INPUT_DIRECTORY, loadFileName);
        eegFeatureAll = load(loadFilePath);

        eegFeature = [];
        eegLabel = [];
        for iTrial = 1: N_TRIALS
            eegFeatureInUse = eegFeatureAll.( ...
                EEG_FEATURE_TYPE + "_" + EEG_SMOOTHING_METHOD + iTrial);
            [nEEGElectrodes, nTimeWindows, nFrequencyBands] = size( ...
                eegFeatureInUse);
            eegFeatureAppend = zeros( ...
                nTimeWindows, nEEGElectrodes * nFrequencyBands);
            for iWindow = 1: nTimeWindows
                 eegFeatureAppend(iWindow, :) = reshape( ...
                     eegFeatureInUse(:, iWindow, :), 1, []);
            end
            eegFeature = [eegFeature; eegFeatureAppend];

             % Non-negative label for classification.
            eegLabelAppend = zeros( ...
                nTimeWindows, 1) + labelOfTrial(iTrial) + 1;
            eegLabel = [eegLabel; eegLabelAppend];
        end

        EEGDataset.feature = eegFeature;
        EEGDataset.label = eegLabel;
        switch iSession
            case 1
                dataset_session1 = EEGDataset;
            case 2
                dataset_session2 = EEGDataset;
            case 3
                dataset_session3 = EEGDataset;
        end

        % Produces string "feature_for_net_session_3_LDS_de" for example.
        saveDirectoryName = join( ...
            [
                "feature_for_net_session" + iSession;
                EEG_SMOOTHING_METHOD;
                EEG_FEATURE_TYPE
            ], ...
            "_" ...
        );
        saveDirectory = fullfile(OUTPUT_DIRECTORY, saveDirectoryName);
        if ~isfolder(saveDirectory), mkdir(saveDirectory); end
        % Produces string "sub_7_session_2.mat" for example.
        saveFileName = ( ...
            "sub_" + iSubject + "_" + "session_" + iSession + ".mat");
        saveFilePath = fullfile(saveDirectory, saveFileName);
        save(saveFilePath, "dataset_session" + iSession)
    end
end
