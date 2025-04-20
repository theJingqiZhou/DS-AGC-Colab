ROOT_DIRECTORY = "data";  % Where every dataset is stored.

DATASET_NAME = "SEED_IV";

EEG_FEATURE_TYPE = "de";  % Differential entropy feature.
EEG_SMOOTHING_METHOD = "";

inputDirectory = fullfile(ROOT_DIRECTORY, DATASET_NAME, "eeg_feature_smooth");
outputDirectory = fullfile( ...
    ROOT_DIRECTORY, DATASET_NAME, EEG_FEATURE_TYPE + "_feature");

N_SUBJECTS = 15;
N_SESSIONS = 3;  % Per subject.
N_TRIALS = 24;  % Per session, per subject.

for iSubject = 1: N_SUBJECTS
    matFilesInfo = dir( ...
        fullfile(inputDirectory, "**", iSubject + "_*.mat"));
    for iSession = 1: N_SESSIONS
        loadFilename = matFilesInfo(iSession).name;
        eegData = load( ...
            fullfile(inputDirectory, num2str(iSession), loadFilename));
        % if j == 1
        %     video_length = [];
        %     for k = 1: n_trials
        %         field_name = 'de_movingAve' + k;
        %         video_length = [video_length;
        %             size(feature_wrap(data_session.(field_name), 1))];
        %     end
        % end
        eegFeature = [];
        eegLabel = [];
        for iTrial = 1: N_TRIALS
        end
    end
end