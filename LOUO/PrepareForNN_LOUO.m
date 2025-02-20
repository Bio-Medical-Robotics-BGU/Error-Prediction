%% This code prepares the kinematic data for inputting into neural networks
%This includes filtering and normalizing the data

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";
DatsetPath = fullfile(project_path, "Datasets_LOUO");
towers = [{'vertical_right'}; {'horizontal_left'}; {'vertical_left'}; {'horizontal_right'}];

%% Designing Filters
%filter for the position (6Hz)
fs = 50;%sampling rate [Hz];
dt = 1/fs;
f_cutoff = 6;%cutoff frequency [Hz]
Wn = f_cutoff/(0.5*fs);%normalized cutoff frequency;
n = 2;%order of lowpass
[b, a] = butter(n, Wn, 'low');%Butterworth filter design

%filter for derivatives (10Hz)
f_cutoff = 10;%cutoff frequency [Hz]
Wn = f_cutoff/(0.5*fs);%normalized cutoff frequency;
n = 2;%order of lowpass
[b10, a10] = butter(n, Wn, 'low');%Butterworth filter design

%% Defining variables
seg_length = 50; %what segment length to take
max_overlap = 0.0;
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

pm = input('1. PSM \n2. MTM \n');

t = input('Which tower would you like? [VR, HL, VL, HR] \n');


tower = towers{t};

%getting relevant indices based on left / right tower
tf = contains(tower, 'right');

if tf
    PSMinds = 1:20;
    MTMinds = 41:60;
else
    PSMinds = 21:40;
    MTMinds = 61:80;
end


if pm == 1
    Inds = PSMinds;
    Tool = 'PSM';
else
    Inds = MTMinds;
    Tool = 'MTM';
end


%running in a loop over all the leave outs
AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
for pp = 1 : length(AllParticipants)

    AllTrainSignalsStandardized = [];

    AllTestSignalsStandardized = [];

    leaveout = AllParticipants(pp);
    disp(leaveout)

    % Loading data
    cd(DatsetPath)
    TrainKinematics = load(['TrainKinematicsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainKinematics;
    TestKinematics = load(['TestKinematicsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestKinematics;

    %All train data
    % Train
    pos = TrainKinematics(:, :, Inds(1):Inds(3));
    pos1 = [];
    for i = 1:size(TrainKinematics, 1)
        x = filtfilt(b,a,pos(i, :, 1));%filtered x
        y = filtfilt(b,a,pos(i, :, 2));%filtered y
        z = filtfilt(b,a,pos(i, :, 3));%filtered z
        pos22 = cat(3, x, y, z);
        pos1 = cat(1, pos1, pos22);
    end

    posx_mean = mean(pos1(:, :, 1), 'all');
    posx_std = std(pos1(:, :, 1), 0, 'all');

    posy_mean = mean(pos1(:, :, 2), 'all');
    posy_std = std(pos1(:, :, 2), 0, 'all');

    posz_mean = mean(pos1(:, :, 3), 'all');
    posz_std = std(pos1(:, :, 3), 0, 'all');

    posx_stand = (pos1(:, :, 1) - posx_mean) / posx_std;
    posy_stand = (pos1(:, :, 2) - posy_mean) / posy_std;
    posz_stand = (pos1(:, :, 3) - posz_mean) / posz_std;
    pos_stand = cat(3, posx_stand, posy_stand, posz_stand);
    assert(mean(pos_stand, "all") - 0 < 1e-10)
    assert(std(pos_stand, 0, "all") - 1 < 1e-10)

    AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, pos_stand);

    %compute velocity
    vel = diff(pos1, 1, 2)./dt;
    vel = cat(2, vel, vel(:, end, :)); %to make same length as rest of signals.
    vel1 = [];
    for i = 1:size(TrainKinematics, 1)
        x = filtfilt(b10,a10,vel(i, :, 1));%filtered x
        y = filtfilt(b10,a10,vel(i, :, 2));%filtered y
        z = filtfilt(b10,a10,vel(i, :, 3));%filtered z
        vel22 = cat(3, x, y, z);
        vel1 = cat(1, vel1, vel22);
    end


    velx_mean = mean(vel1(:, :, 1), 'all');
    velx_std = std(vel1(:, :, 1), 0, 'all');

    vely_mean = mean(vel1(:, :, 2), 'all');
    vely_std = std(vel1(:, :, 2), 0, 'all');

    velz_mean = mean(vel1(:, :, 3), 'all');
    velz_std = std(vel1(:, :, 3), 0, 'all');

    velx_stand = (vel1(:, :, 1) - velx_mean) / velx_std;
    vely_stand = (vel1(:, :, 2) - vely_mean) / vely_std;
    velz_stand = (vel1(:, :, 3) - velz_mean) / velz_std;
    vel_stand = cat(3, velx_stand, vely_stand, velz_stand);
    assert(mean(vel_stand, "all") - 0 < 1e-10)
    assert(std(vel_stand, 0, "all") - 1 < 1e-10)

    AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, vel_stand);


    % Angular Velocity
    angvel = TrainKinematics(:, :, Inds(7):Inds(9));
    angvel1 = [];
    for i = 1:size(TrainKinematics, 1)
        x = filtfilt(b10,a10,angvel(i, :, 1));%filtered x
        y = filtfilt(b10,a10,angvel(i, :, 2));%filtered y
        z = filtfilt(b10,a10,angvel(i, :, 3));%filtered z
        vel22 = cat(3, x, y, z);
        angvel1 = cat(1, angvel1, vel22);
    end

    angvelx_mean = mean(angvel1(:, :, 1), 'all');
    angvelx_std = std(angvel1(:, :, 1), 0, 'all');

    angvely_mean = mean(angvel1(:, :, 2), 'all');
    angvely_std = std(angvel1(:, :, 2), 0, 'all');

    angvelz_mean = mean(angvel1(:, :, 3), 'all');
    angvelz_std = std(angvel1(:, :, 3), 0, 'all');


    angvelx_stand = (angvel1(:, :, 1) - angvelx_mean) / angvelx_std;
    angvely_stand = (angvel1(:, :, 2) - angvely_mean) / angvely_std;
    angvelz_stand = (angvel1(:, :, 3) - angvelz_mean) / angvelz_std;
    angvel_stand = cat(3, angvelx_stand, angvely_stand, angvelz_stand);
    assert(mean(angvel_stand, "all") - 0 < 1e-10)
    assert(std(angvel_stand, 0, "all") - 1 < 1e-10)

    AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, angvel_stand);

    % Orientation
    orientations = TrainKinematics(:, :, Inds(17):Inds(20));

    or1_mean = mean(orientations(:, :, 1), 'all');
    or1_std = std(orientations(:, :, 1), 0, 'all');

    or2_mean = mean(orientations(:, :, 2), 'all');
    or2_std = std(orientations(:, :, 2), 0, 'all');

    or3_mean = mean(orientations(:, :, 3), 'all');
    or3_std = std(orientations(:, :, 3), 0, 'all');

    or4_mean = mean(orientations(:, :, 4), 'all');
    or4_std = std(orientations(:, :, 4), 0, 'all');


    or1_stand = (orientations(:, :, 1) - or1_mean) / or1_std;
    or2_stand = (orientations(:, :, 2) - or2_mean) / or2_std;
    or3_stand = (orientations(:, :, 3) - or3_mean) / or3_std;
    or4_stand = (orientations(:, :, 4) - or4_mean) / or4_std;
    or_stand = cat(3, or1_stand, or2_stand, or3_stand, or4_stand);
    assert(mean(or_stand, "all") - 0 < 1e-10)
    assert(std(or_stand, 0, "all") - 1 < 1e-10)

    AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, or_stand);

    cd(DatsetPath)
    save(['TrainSignalsStandardized_LOUO_', leaveout, '_',  Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsStandardized');


    % Tests
    pos = TestKinematics(:, :, Inds(1):Inds(3));
    pos1 = [];
    for i = 1:size(TestKinematics, 1)
        x = filtfilt(b,a,pos(i, :, 1));%filtered x
        y = filtfilt(b,a,pos(i, :, 2));%filtered y
        z = filtfilt(b,a,pos(i, :, 3));%filtered z
        pos22 = cat(3, x, y, z);
        pos1 = cat(1, pos1, pos22);
    end

    posx_stand = (pos1(:, :, 1) - posx_mean) / posx_std;
    posy_stand = (pos1(:, :, 2) - posy_mean) / posy_std;
    posz_stand = (pos1(:, :, 3) - posz_mean) / posz_std;
    pos_stand = cat(3, posx_stand, posy_stand, posz_stand);

    AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, pos_stand);

    %compute velocity
    vel = diff(pos1, 1, 2)./dt;
    vel = cat(2, vel, vel(:, end, :)); %to make same length as rest of signals.
    vel1 = [];
    for i = 1:size(TestKinematics, 1)
        x = filtfilt(b10,a10,vel(i, :, 1));%filtered x
        y = filtfilt(b10,a10,vel(i, :, 2));%filtered y
        z = filtfilt(b10,a10,vel(i, :, 3));%filtered z
        vel22 = cat(3, x, y, z);
        vel1 = cat(1, vel1, vel22);
    end

    velx_stand = (vel1(:, :, 1) - velx_mean) / velx_std;
    vely_stand = (vel1(:, :, 2) - vely_mean) / vely_std;
    velz_stand = (vel1(:, :, 3) - velz_mean) / velz_std;
    vel_stand = cat(3, velx_stand, vely_stand, velz_stand);

    AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, vel_stand);

    % Angular Velocity
    angvel = TestKinematics(:, :, Inds(7):Inds(9));
    angvel1 = [];
    for i = 1:size(TestKinematics, 1)
        x = filtfilt(b10,a10,angvel(i, :, 1));%filtered x
        y = filtfilt(b10,a10,angvel(i, :, 2));%filtered y
        z = filtfilt(b10,a10,angvel(i, :, 3));%filtered z
        vel22 = cat(3, x, y, z);
        angvel1 = cat(1, angvel1, vel22);
    end

    angvelx_stand = (angvel1(:, :, 1) - angvelx_mean) / angvelx_std;
    angvely_stand = (angvel1(:, :, 2) - angvely_mean) / angvely_std;
    angvelz_stand = (angvel1(:, :, 3) - angvelz_mean) / angvelz_std;
    angvel_stand = cat(3, angvelx_stand, angvely_stand, angvelz_stand);

    AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, angvel_stand);

    % Orientation
    orientations = TestKinematics(:, :, Inds(17):Inds(20));

    or1_stand = (orientations(:, :, 1) - or1_mean) / or1_std;
    or2_stand = (orientations(:, :, 2) - or2_mean) / or2_std;
    or3_stand = (orientations(:, :, 3) - or3_mean) / or3_std;
    or4_stand = (orientations(:, :, 4) - or4_mean) / or4_std;
    or_stand = cat(3, or1_stand, or2_stand, or3_stand, or4_stand);

    AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, or_stand);

    cd(DatsetPath)
    save(['TestSignalsStandardized_LOUO_', leaveout, '_',  Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsStandardized');

end %end of running over all participants

%% Combine the four towers
project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";
DatsetPath = fullfile(project_path, "Datasets_LOUO");

seg_length = 50; %what segment length to take
max_overlap = 0;
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 100 hz)
Tool = 'PSM';

cd(DatsetPath)

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
for pp = 1 : length(AllParticipants)
    leaveout = AllParticipants(pp);
    disp(leaveout)

    %train
    TrainSignalsStandardized1 = load(['TrainSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

    TrainSignalsStandardized2 = load(['TrainSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

    TrainSignalsStandardized3 = load(['TrainSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

    TrainSignalsStandardized4 = load(['TrainSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

    AllTrainSignalsStandardized = cat(1, TrainSignalsStandardized1, TrainSignalsStandardized2, TrainSignalsStandardized3, TrainSignalsStandardized4);

    save(['AllTrainSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsStandardized');


    %test
    TestSignalsStandardized1 = load(['TestSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

    TestSignalsStandardized2 = load(['TestSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

    TestSignalsStandardized3 = load(['TestSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

    TestSignalsStandardized4 = load(['TestSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

    AllTestSignalsStandardized = cat(1, TestSignalsStandardized1, TestSignalsStandardized2, TestSignalsStandardized3, TestSignalsStandardized4);

    save(['AllTestSignalsStandardized_LOUO_', leaveout, '_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsStandardized');

end

%% Concatenate Labels
for pp = 1 : length(AllParticipants)
    leaveout = AllParticipants(pp);
    disp(leaveout)

    TrainLabels1 = load(['TrainLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
    TrainLabels2 = load(['TrainLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
    TrainLabels3 = load(['TrainLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
    TrainLabels4 = load(['TrainLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
    AllTrainLabels = cat(1, TrainLabels1, TrainLabels2, TrainLabels3, TrainLabels4);
    save(['AllTrainLabels_LOUO_', leaveout, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainLabels');

    TestLabels1 = load(['TestLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(0*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
    TestLabels2 = load(['TestLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(0*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
    TestLabels3 = load(['TestLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(0*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
    TestLabels4 = load(['TestLabelsElimOverlapNoMix_LOUO_', leaveout, '_', num2str(0*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
    AllTestLabels = cat(1, TestLabels1, TestLabels2, TestLabels3, TestLabels4);
    save(['AllTestLabels_LOUO_', leaveout, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestLabels');

end

%% Checking class weight
AllClassWeights = zeros(length(AllParticipants), 1);
for pp = 1 : length(AllParticipants)
    leaveout = AllParticipants(pp);
    disp(leaveout)
    AllTrainLabels = load(['AllTrainLabels_LOUO_', leaveout, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainLabels;

    AllClassWeights(pp) = length(find(AllTrainLabels == 0)) / sum(AllTrainLabels);
end
