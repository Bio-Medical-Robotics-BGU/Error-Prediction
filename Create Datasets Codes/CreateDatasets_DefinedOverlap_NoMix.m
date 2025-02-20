%% This code creates the kinematic matrices and labels of the train, validation
% and test sets. We can set the chosen overlap, while the overlap does
% not allow for overlap between segments of different classes.

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";

DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
SavePath = fullfile(project_path, "DatasetsTrainValTest_OneSplit");

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
AllMonths = ['1', '2', '3', '4', '5', '6'];
AllTimes = ['a', 'b', 'c'];
towers = [{'vertical_right'}; {'horizontal_left'}; {'vertical_left'}; {'horizontal_right'}];

%% Defining variables

tower = towers{4};

seg_length1 = 50; %for loading tables
max_overlap = 0; 
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

%% new overlap
seg_length = 50; %what segment length to take
max_overlap2 = 0;
advance2 = 1; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

%% Creating summary tables for each dataset
for pp = 1:length(AllParticipants)
    participant = AllParticipants(pp);
    disp(participant)

    cd(SavePath)
    try
        TrainTable = load(['TrainTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length1), '_ad', num2str(advance), '_50Hz.mat']).data;
    catch
        TrainTable = [];
    end
    try
        ValTable = load(['ValTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length1), '_ad', num2str(advance), '_50Hz.mat']).data;
    catch
        ValTable = [];
    end
    try
        TestTable = load(['TestTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length1), '_ad', num2str(advance), '_50Hz.mat']).data;
    catch
        TestTable = [];
    end
    if pp == 1
        AllTrains = TrainTable;
        AllVals = ValTable;
        AllTests = TestTable;
    else
        AllTrains = [AllTrains; TrainTable];
        AllVals = [AllVals; ValTable];
        AllTests = [AllTests; TestTable];
    end

end %end of running over participants

%% Train
cd(DataPath)
NumTrainErr = 0;
NumTrainNot = 0;
TrainLabels = [];
TrainSegmentLabels = [];
TrainKinematics = [];

for i = 1:height(AllTrains)
    participant = table2array(AllTrains(i, "Participant")); participant = participant{1};
    month = num2str(table2array(AllTrains(i, "Month")));
    tt = table2array(AllTrains(i, "Session")); tt = tt{1};
    try
        MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).MoveLabel;
        TowerSegmentLabel = load(['TowerSegmentLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerSegmentLabel;
        TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerKinematics;
    catch
        continue
    end
    oness = length(find(MoveLabel == 1));
    zeross = length(find(MoveLabel == 0));
    % assert (oness + zeross == length(MoveLabel))
    % assert(oness == table2array(AllTrains(i, "Num Before Error")))
    % assert(size(TowerKinematics, 1) == length(MoveLabel))

    NumTrainErr = NumTrainErr + oness;
    NumTrainNot = NumTrainNot + zeross;

    TrainLabels = [TrainLabels; MoveLabel];
    TrainSegmentLabels = [TrainSegmentLabels; TowerSegmentLabel];
    TrainKinematics = cat(1, TrainKinematics, TowerKinematics);

end
cd(SavePath)
save(['TrainKinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TrainKinematics')
save(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TrainLabels')
save(['TrainSegmentLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TrainSegmentLabels')
               
%% Val
cd(DataPath)
NumValErr = 0;
NumValNot = 0;
ValLabels = [];
ValSegmentLabels = [];
ValKinematics = [];

for i = 1:height(AllVals)
    participant = table2array(AllVals(i, "Participant")); participant = participant{1};
    month = num2str(table2array(AllVals(i, "Month")));
    tt = table2array(AllVals(i, "Session")); tt = tt{1};
    try
        MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).MoveLabel;
        TowerSegmentLabel = load(['TowerSegmentLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerSegmentLabel;
        TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerKinematics;
    catch
        continue
    end
    oness = length(find(MoveLabel == 1));
    zeross = length(find(MoveLabel == 0));
    % assert (oness + zeross == length(MoveLabel))
    % assert(oness == table2array(AllVals(i, "Num Before Error")))
    % assert(size(TowerKinematics, 1) == length(MoveLabel))

    NumValErr = NumValErr + oness;
    NumValNot = NumValNot + zeross;

    ValLabels = [ValLabels; MoveLabel];
    ValSegmentLabels = [ValSegmentLabels; TowerSegmentLabel];
    ValKinematics = cat(1, ValKinematics, TowerKinematics);

end
cd(SavePath)
save(['ValKinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'ValKinematics')
save(['ValLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'ValLabels')
save(['ValSegmentLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'ValSegmentLabels')
      
%% Test
cd(DataPath)
NumTestErr = 0;
NumTestNot = 0;
TestLabels = [];
TestSegmentLabels = [];
TestKinematics = [];

for i = 1:height(AllTests)
    participant = table2array(AllTests(i, "Participant")); participant = participant{1};
    month = num2str(table2array(AllTests(i, "Month")));
    tt = table2array(AllTests(i, "Session")); tt = tt{1};
    try
        MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).MoveLabel;
        TowerSegmentLabel = load(['TowerSegmentLabelElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerSegmentLabel;
        TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz', '.mat']).TowerKinematics;
    catch
        continue
    end
    oness = length(find(MoveLabel == 1));
    zeross = length(find(MoveLabel == 0));
    % assert (oness + zeross == length(MoveLabel))
    % assert(oness == table2array(AllTests(i, "Num Before Error")))
    % assert(size(TowerKinematics, 1) == length(MoveLabel))

    NumTestErr = NumTestErr + oness;
    NumTestNot = NumTestNot + zeross;

    TestLabels = [TestLabels; MoveLabel];
    TestSegmentLabels = [TestSegmentLabels; TowerSegmentLabel];
    TestKinematics = cat(1, TestKinematics, TowerKinematics);

end
cd(SavePath)
save(['TestKinematicsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TestKinematics')
save(['TestLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TestLabels')
save(['TestSegmentLabelsElimOverlapNoMix_', num2str(max_overlap2*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance2), '_50Hz'], 'TestSegmentLabels')
        