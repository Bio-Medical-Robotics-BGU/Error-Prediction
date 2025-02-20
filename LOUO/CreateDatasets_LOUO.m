%% This code creates the kinematic matrices and labels of the train and test sets using Leave
% One User Out

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";

DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
SavePath = fullfile(project_path, "Datasets_LOUO");

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
AllMonths = ['1', '2', '3', '4', '5', '6'];
AllTimes = ['a', 'b', 'c'];
towers = [{'vertical_right'}; {'horizontal_left'}; {'vertical_left'}; {'horizontal_right'}];

%% Defining variables
tower = towers{4};

seg_length = 50;
max_overlap = 0;
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

%%

for i = 1 : length(AllParticipants)

    leaveout = AllParticipants(i);

    participants = setdiff(AllParticipants, leaveout); %all the training participants for this leave out

    disp(['leave out is: ',  leaveout])
    disp(['included are: ',  participants])

    TrainLabels = [];
    TrainKinematics = [];

    TestLabels = [];
    TestKinematics = [];


    cd(DataPath)

    for pp = 1 : length(participants)
        participant = participants(pp);

        for mm = 1:length(AllMonths)
            month = AllMonths(mm);

            for ttt = 1:length(AllTimes)
                tt = AllTimes(ttt);

                try
                    MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;
                    TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).TowerKinematics;
                catch
                    continue
                end

                TrainLabels = [TrainLabels; MoveLabel];
                TrainKinematics = cat(1, TrainKinematics, TowerKinematics);

            end %end of running over sessions
        end %end of running over months


    end %end of running over included participants

    %save data of leave out
    participant = leaveout;

    for mm = 1:length(AllMonths)
        month = AllMonths(mm);

        for ttt = 1:length(AllTimes)
            tt = AllTimes(ttt);

            try
                MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;
                TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).TowerKinematics;
            catch
                continue
            end

            TestLabels = [TestLabels; MoveLabel];
            TestKinematics = cat(1, TestKinematics, TowerKinematics);

        end %end of running over sessions
    end %end of running over months

    cd(SavePath)
    save(['TrainKinematicsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TrainKinematics')
    save(['TrainLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TrainLabels')

    save(['TestKinematicsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TestKinematics')
    save(['TestLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TestLabels')


end %end of leaving out each participant in turn

