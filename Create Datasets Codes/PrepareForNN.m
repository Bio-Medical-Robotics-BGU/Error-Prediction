%% This code prepares the kinematic data for inputting into neural networks
%This includes filtering and normalizing the data

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";
DatsetPath = fullfile(project_path, "DatasetsTrainValTest_OneSplit");
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
advance = 25; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

pm = input('1. PSM \n2. MTM \n');

t = input('Which tower would you like? [VR, HL, VL, HR] \n');

AllTrainSignalsNormalized = [];
AllTrainSignalsStandardized = [];

AllValSignalsNormalized = [];
AllValSignalsStandardized = [];

AllTestSignalsNormalized = [];
AllTestSignalsStandardized = [];

% Loading data
cd(DatsetPath)

tower = towers{t};
disp(tower)

TrainKinematics = load(['TrainKinematicsElimOverlapNoMix_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainKinematics;
% TrainLabels = load(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;

ValKinematics = load(['ValKinematicsElimOverlapNoMix_', num2str(0*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValKinematics;
% ValLabels = load(['ValLabelsElimOverlapNoMix_', num2str(0*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValLabels;

TestKinematics = load(['TestKinematicsElimOverlapNoMix_', num2str(0*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestKinematics;
% TestLabels = load(['TestLabelsElimOverlapNoMix_', num2str(0*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;

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

posx_min = min(pos1(:, :, 1), [], 'all');
posx_max = max(pos1(:, :, 1), [], 'all');

posy_min = min(pos1(:, :, 2), [], 'all');
posy_max = max(pos1(:, :, 2), [], 'all');

posz_min = min(pos1(:, :, 3), [], 'all');
posz_max = max(pos1(:, :, 3), [], 'all');


posx_mean = mean(pos1(:, :, 1), 'all');
posx_std = std(pos1(:, :, 1), 0, 'all');

posy_mean = mean(pos1(:, :, 2), 'all');
posy_std = std(pos1(:, :, 2), 0, 'all');

posz_mean = mean(pos1(:, :, 3), 'all');
posz_std = std(pos1(:, :, 3), 0, 'all');

posx_norm = (pos1(:, :, 1) - posx_min) / (posx_max - posx_min);
posy_norm = (pos1(:, :, 2) - posy_min) / (posy_max - posy_min);
posz_norm = (pos1(:, :, 3) - posz_min) / (posz_max - posz_min);
pos_norm = cat(3, posx_norm, posy_norm, posz_norm);
assert(min(pos_norm, [], "all") == 0)
assert(max(pos_norm, [], "all") == 1)

posx_stand = (pos1(:, :, 1) - posx_mean) / posx_std;
posy_stand = (pos1(:, :, 2) - posy_mean) / posy_std;
posz_stand = (pos1(:, :, 3) - posz_mean) / posz_std;
pos_stand = cat(3, posx_stand, posy_stand, posz_stand);
assert(mean(pos_stand, "all") - 0 < 1e-10)
assert(std(pos_stand, 0, "all") - 1 < 1e-10)

AllTrainSignalsNormalized = cat(3, AllTrainSignalsNormalized, pos_norm);
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

velx_min = min(vel1(:, :, 1), [], 'all');
velx_max = max(vel1(:, :, 1), [], 'all');

vely_min = min(vel1(:, :, 2), [], 'all');
vely_max = max(vel1(:, :, 2), [], 'all');

velz_min = min(vel1(:, :, 3), [], 'all');
velz_max = max(vel1(:, :, 3), [], 'all');


velx_mean = mean(vel1(:, :, 1), 'all');
velx_std = std(vel1(:, :, 1), 0, 'all');

vely_mean = mean(vel1(:, :, 2), 'all');
vely_std = std(vel1(:, :, 2), 0, 'all');

velz_mean = mean(vel1(:, :, 3), 'all');
velz_std = std(vel1(:, :, 3), 0, 'all');

velx_norm = (vel1(:, :, 1) - velx_min) / (velx_max - velx_min);
vely_norm = (vel1(:, :, 2) - vely_min) / (vely_max - vely_min);
velz_norm = (vel1(:, :, 3) - velz_min) / (velz_max - velz_min);
vel_norm = cat(3, velx_norm, vely_norm, velz_norm);
assert(min(vel_norm, [], "all") == 0)
assert(max(vel_norm, [], "all") == 1)

velx_stand = (vel1(:, :, 1) - velx_mean) / velx_std;
vely_stand = (vel1(:, :, 2) - vely_mean) / vely_std;
velz_stand = (vel1(:, :, 3) - velz_mean) / velz_std;
vel_stand = cat(3, velx_stand, vely_stand, velz_stand);
assert(mean(vel_stand, "all") - 0 < 1e-10)
assert(std(vel_stand, 0, "all") - 1 < 1e-10)

AllTrainSignalsNormalized = cat(3, AllTrainSignalsNormalized, vel_norm);
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

angvelx_min = min(angvel1(:, :, 1), [], 'all');
angvelx_max = max(angvel1(:, :, 1), [], 'all');

angvely_min = min(angvel1(:, :, 2), [], 'all');
angvely_max = max(angvel1(:, :, 2), [], 'all');

angvelz_min = min(angvel1(:, :, 3), [], 'all');
angvelz_max = max(angvel1(:, :, 3), [], 'all');


angvelx_mean = mean(angvel1(:, :, 1), 'all');
angvelx_std = std(angvel1(:, :, 1), 0, 'all');

angvely_mean = mean(angvel1(:, :, 2), 'all');
angvely_std = std(angvel1(:, :, 2), 0, 'all');

angvelz_mean = mean(angvel1(:, :, 3), 'all');
angvelz_std = std(angvel1(:, :, 3), 0, 'all');

angvelx_norm = (angvel1(:, :, 1) - angvelx_min) / (angvelx_max - angvelx_min);
angvely_norm = (angvel1(:, :, 2) - angvely_min) / (angvely_max - angvely_min);
angvelz_norm = (angvel1(:, :, 3) - angvelz_min) / (angvelz_max - angvelz_min);
angvel_norm = cat(3, angvelx_norm, angvely_norm, angvelz_norm);
assert(min(angvel_norm, [], "all") == 0)
assert(max(angvel_norm, [], "all") == 1)

angvelx_stand = (angvel1(:, :, 1) - angvelx_mean) / angvelx_std;
angvely_stand = (angvel1(:, :, 2) - angvely_mean) / angvely_std;
angvelz_stand = (angvel1(:, :, 3) - angvelz_mean) / angvelz_std;
angvel_stand = cat(3, angvelx_stand, angvely_stand, angvelz_stand);
assert(mean(angvel_stand, "all") - 0 < 1e-10)
assert(std(angvel_stand, 0, "all") - 1 < 1e-10)

AllTrainSignalsNormalized = cat(3, AllTrainSignalsNormalized, angvel_norm);
AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, angvel_stand);

% Orientation
orientations = TrainKinematics(:, :, Inds(17):Inds(20));

or1_min = min(orientations(:, :, 1), [], 'all');
or1_max = max(orientations(:, :, 1), [], 'all');

or2_min = min(orientations(:, :, 2), [], 'all');
or2_max = max(orientations(:, :, 2), [], 'all');

or3_min = min(orientations(:, :, 3), [], 'all');
or3_max = max(orientations(:, :, 3), [], 'all');

or4_min = min(orientations(:, :, 4), [], 'all');
or4_max = max(orientations(:, :, 4), [], 'all');


or1_mean = mean(orientations(:, :, 1), 'all');
or1_std = std(orientations(:, :, 1), 0, 'all');

or2_mean = mean(orientations(:, :, 2), 'all');
or2_std = std(orientations(:, :, 2), 0, 'all');

or3_mean = mean(orientations(:, :, 3), 'all');
or3_std = std(orientations(:, :, 3), 0, 'all');

or4_mean = mean(orientations(:, :, 4), 'all');
or4_std = std(orientations(:, :, 4), 0, 'all');

or1_norm = (orientations(:, :, 1) - or1_min) / (or1_max - or1_min);
or2_norm = (orientations(:, :, 2) - or2_min) / (or2_max - or2_min);
or3_norm = (orientations(:, :, 3) - or3_min) / (or3_max - or3_min);
or4_norm = (orientations(:, :, 4) - or4_min) / (or4_max - or4_min);
or_norm = cat(3, or1_norm, or2_norm, or3_norm, or4_norm);
assert(min(or_norm, [], "all") == 0)
assert(max(or_norm, [], "all") == 1)

or1_stand = (orientations(:, :, 1) - or1_mean) / or1_std;
or2_stand = (orientations(:, :, 2) - or2_mean) / or2_std;
or3_stand = (orientations(:, :, 3) - or3_mean) / or3_std;
or4_stand = (orientations(:, :, 4) - or4_mean) / or4_std;
or_stand = cat(3, or1_stand, or2_stand, or3_stand, or4_stand);
assert(mean(or_stand, "all") - 0 < 1e-10)
assert(std(or_stand, 0, "all") - 1 < 1e-10)

AllTrainSignalsNormalized = cat(3, AllTrainSignalsNormalized, or_norm);
AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, or_stand);

% Jaw
jaw = TrainKinematics(:, :, Inds(16));

jaw_min = min(jaw, [], 'all');
jaw_max = max(jaw, [], 'all');

jaw_mean = mean(jaw, 'all');
jaw_std = std(jaw, 0, 'all');

jaw_norm = (jaw - jaw_min) / (jaw_max - jaw_min);
assert(min(jaw_norm, [], "all") == 0)
assert(max(jaw_norm, [], "all") == 1)

jaw_stand = (jaw - jaw_mean) / jaw_std;
assert(mean(jaw_stand, "all") - 0 < 1e-10)
assert(std(jaw_stand, 0, "all") - 1 < 1e-10)

AllTrainSignalsNormalized = cat(3, AllTrainSignalsNormalized, jaw_norm);
AllTrainSignalsStandardized = cat(3, AllTrainSignalsStandardized, jaw_stand);

cd(DatsetPath)
save(['TrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsNormalized');
save(['TrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsStandardized');


% Validation
pos = ValKinematics(:, :, Inds(1):Inds(3));
pos1 = [];
for i = 1:size(ValKinematics, 1)
    x = filtfilt(b,a,pos(i, :, 1));%filtered x
    y = filtfilt(b,a,pos(i, :, 2));%filtered y
    z = filtfilt(b,a,pos(i, :, 3));%filtered z
    pos22 = cat(3, x, y, z);
    pos1 = cat(1, pos1, pos22);
end

posx_norm = (pos1(:, :, 1) - posx_min) / (posx_max - posx_min);
posy_norm = (pos1(:, :, 2) - posy_min) / (posy_max - posy_min);
posz_norm = (pos1(:, :, 3) - posz_min) / (posz_max - posz_min);
pos_norm = cat(3, posx_norm, posy_norm, posz_norm);


posx_stand = (pos1(:, :, 1) - posx_mean) / posx_std;
posy_stand = (pos1(:, :, 2) - posy_mean) / posy_std;
posz_stand = (pos1(:, :, 3) - posz_mean) / posz_std;
pos_stand = cat(3, posx_stand, posy_stand, posz_stand);

AllValSignalsNormalized = cat(3, AllValSignalsNormalized, pos_norm);
AllValSignalsStandardized = cat(3, AllValSignalsStandardized, pos_stand);

%compute velocity
vel = diff(pos1, 1, 2)./dt;
vel = cat(2, vel, vel(:, end, :)); %to make same length as rest of signals.
vel1 = [];
for i = 1:size(ValKinematics, 1)
    x = filtfilt(b10,a10,vel(i, :, 1));%filtered x
    y = filtfilt(b10,a10,vel(i, :, 2));%filtered y
    z = filtfilt(b10,a10,vel(i, :, 3));%filtered z
    vel22 = cat(3, x, y, z);
    vel1 = cat(1, vel1, vel22);
end

velx_norm = (vel1(:, :, 1) - velx_min) / (velx_max - velx_min);
vely_norm = (vel1(:, :, 2) - vely_min) / (vely_max - vely_min);
velz_norm = (vel1(:, :, 3) - velz_min) / (velz_max - velz_min);
vel_norm = cat(3, velx_norm, vely_norm, velz_norm);

velx_stand = (vel1(:, :, 1) - velx_mean) / velx_std;
vely_stand = (vel1(:, :, 2) - vely_mean) / vely_std;
velz_stand = (vel1(:, :, 3) - velz_mean) / velz_std;
vel_stand = cat(3, velx_stand, vely_stand, velz_stand);

AllValSignalsNormalized = cat(3, AllValSignalsNormalized, vel_norm);
AllValSignalsStandardized = cat(3, AllValSignalsStandardized, vel_stand);


% Angular Velocity
angvel = ValKinematics(:, :, Inds(7):Inds(9));
angvel1 = [];
for i = 1:size(ValKinematics, 1)
    x = filtfilt(b10,a10,angvel(i, :, 1));%filtered x
    y = filtfilt(b10,a10,angvel(i, :, 2));%filtered y
    z = filtfilt(b10,a10,angvel(i, :, 3));%filtered z
    vel22 = cat(3, x, y, z);
    angvel1 = cat(1, angvel1, vel22);
end

angvelx_norm = (angvel1(:, :, 1) - angvelx_min) / (angvelx_max - angvelx_min);
angvely_norm = (angvel1(:, :, 2) - angvely_min) / (angvely_max - angvely_min);
angvelz_norm = (angvel1(:, :, 3) - angvelz_min) / (angvelz_max - angvelz_min);
angvel_norm = cat(3, angvelx_norm, angvely_norm, angvelz_norm);

angvelx_stand = (angvel1(:, :, 1) - angvelx_mean) / angvelx_std;
angvely_stand = (angvel1(:, :, 2) - angvely_mean) / angvely_std;
angvelz_stand = (angvel1(:, :, 3) - angvelz_mean) / angvelz_std;
angvel_stand = cat(3, angvelx_stand, angvely_stand, angvelz_stand);

AllValSignalsNormalized = cat(3, AllValSignalsNormalized, angvel_norm);
AllValSignalsStandardized = cat(3, AllValSignalsStandardized, angvel_stand);

% Orientation
orientations = ValKinematics(:, :, Inds(17):Inds(20));

or1_norm = (orientations(:, :, 1) - or1_min) / (or1_max - or1_min);
or2_norm = (orientations(:, :, 2) - or2_min) / (or2_max - or2_min);
or3_norm = (orientations(:, :, 3) - or3_min) / (or3_max - or3_min);
or4_norm = (orientations(:, :, 4) - or4_min) / (or4_max - or4_min);
or_norm = cat(3, or1_norm, or2_norm, or3_norm, or4_norm);

or1_stand = (orientations(:, :, 1) - or1_mean) / or1_std;
or2_stand = (orientations(:, :, 2) - or2_mean) / or2_std;
or3_stand = (orientations(:, :, 3) - or3_mean) / or3_std;
or4_stand = (orientations(:, :, 4) - or4_mean) / or4_std;
or_stand = cat(3, or1_stand, or2_stand, or3_stand, or4_stand);

AllValSignalsNormalized = cat(3, AllValSignalsNormalized, or_norm);
AllValSignalsStandardized = cat(3, AllValSignalsStandardized, or_stand);

% Jaw
jaw = ValKinematics(:, :, Inds(16));

jaw_norm = (jaw - jaw_min) / (jaw_max - jaw_min);
jaw_stand = (jaw - jaw_mean) / jaw_std;

AllValSignalsNormalized = cat(3, AllValSignalsNormalized, jaw_norm);
AllValSignalsStandardized = cat(3, AllValSignalsStandardized, jaw_stand);

cd(DatsetPath)
save(['ValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValSignalsNormalized');
save(['ValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValSignalsStandardized');

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

posx_norm = (pos1(:, :, 1) - posx_min) / (posx_max - posx_min);
posy_norm = (pos1(:, :, 2) - posy_min) / (posy_max - posy_min);
posz_norm = (pos1(:, :, 3) - posz_min) / (posz_max - posz_min);
pos_norm = cat(3, posx_norm, posy_norm, posz_norm);

posx_stand = (pos1(:, :, 1) - posx_mean) / posx_std;
posy_stand = (pos1(:, :, 2) - posy_mean) / posy_std;
posz_stand = (pos1(:, :, 3) - posz_mean) / posz_std;
pos_stand = cat(3, posx_stand, posy_stand, posz_stand);

AllTestSignalsNormalized = cat(3, AllTestSignalsNormalized, pos_norm);
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

velx_norm = (vel1(:, :, 1) - velx_min) / (velx_max - velx_min);
vely_norm = (vel1(:, :, 2) - vely_min) / (vely_max - vely_min);
velz_norm = (vel1(:, :, 3) - velz_min) / (velz_max - velz_min);
vel_norm = cat(3, velx_norm, vely_norm, velz_norm);

velx_stand = (vel1(:, :, 1) - velx_mean) / velx_std;
vely_stand = (vel1(:, :, 2) - vely_mean) / vely_std;
velz_stand = (vel1(:, :, 3) - velz_mean) / velz_std;
vel_stand = cat(3, velx_stand, vely_stand, velz_stand);

AllTestSignalsNormalized = cat(3, AllTestSignalsNormalized, vel_norm);
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

angvelx_norm = (angvel1(:, :, 1) - angvelx_min) / (angvelx_max - angvelx_min);
angvely_norm = (angvel1(:, :, 2) - angvely_min) / (angvely_max - angvely_min);
angvelz_norm = (angvel1(:, :, 3) - angvelz_min) / (angvelz_max - angvelz_min);
angvel_norm = cat(3, angvelx_norm, angvely_norm, angvelz_norm);

angvelx_stand = (angvel1(:, :, 1) - angvelx_mean) / angvelx_std;
angvely_stand = (angvel1(:, :, 2) - angvely_mean) / angvely_std;
angvelz_stand = (angvel1(:, :, 3) - angvelz_mean) / angvelz_std;
angvel_stand = cat(3, angvelx_stand, angvely_stand, angvelz_stand);

AllTestSignalsNormalized = cat(3, AllTestSignalsNormalized, angvel_norm);
AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, angvel_stand);

% Orientation
orientations = TestKinematics(:, :, Inds(17):Inds(20));

or1_norm = (orientations(:, :, 1) - or1_min) / (or1_max - or1_min);
or2_norm = (orientations(:, :, 2) - or2_min) / (or2_max - or2_min);
or3_norm = (orientations(:, :, 3) - or3_min) / (or3_max - or3_min);
or4_norm = (orientations(:, :, 4) - or4_min) / (or4_max - or4_min);
or_norm = cat(3, or1_norm, or2_norm, or3_norm, or4_norm);

or1_stand = (orientations(:, :, 1) - or1_mean) / or1_std;
or2_stand = (orientations(:, :, 2) - or2_mean) / or2_std;
or3_stand = (orientations(:, :, 3) - or3_mean) / or3_std;
or4_stand = (orientations(:, :, 4) - or4_mean) / or4_std;
or_stand = cat(3, or1_stand, or2_stand, or3_stand, or4_stand);

AllTestSignalsNormalized = cat(3, AllTestSignalsNormalized, or_norm);
AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, or_stand);

% Jaw
jaw = TestKinematics(:, :, Inds(16));

jaw_norm = (jaw - jaw_min) / (jaw_max - jaw_min);
jaw_stand = (jaw - jaw_mean) / jaw_std;

AllTestSignalsNormalized = cat(3, AllTestSignalsNormalized, jaw_norm);
AllTestSignalsStandardized = cat(3, AllTestSignalsStandardized, jaw_stand);

cd(DatsetPath)
save(['TestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsNormalized');
save(['TestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsStandardized');

%% Combine the four towers
seg_length = 50; %what segment length to take
max_overlap = 0;
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 100 hz)
Tool = 'PSM';

cd(DatsetPath)
%train
TrainSignalsNormalized1 = load(['TrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsNormalized;
TrainSignalsStandardized1 = load(['TrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

TrainSignalsNormalized2 = load(['TrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsNormalized;
TrainSignalsStandardized2 = load(['TrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

TrainSignalsNormalized3 = load(['TrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsNormalized;
TrainSignalsStandardized3 = load(['TrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

TrainSignalsNormalized4 = load(['TrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsNormalized;
TrainSignalsStandardized4 = load(['TrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTrainSignalsStandardized;

AllTrainSignalsNormalized = cat(1, TrainSignalsNormalized1, TrainSignalsNormalized2, TrainSignalsNormalized3, TrainSignalsNormalized4);
AllTrainSignalsStandardized = cat(1, TrainSignalsStandardized1, TrainSignalsStandardized2, TrainSignalsStandardized3, TrainSignalsStandardized4);

save(['AllTrainSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsNormalized');
save(['AllTrainSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSignalsStandardized');

%val
ValSignalsNormalized1 = load(['ValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsNormalized;
ValSignalsStandardized1 = load(['ValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsStandardized;

ValSignalsNormalized2 = load(['ValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsNormalized;
ValSignalsStandardized2 = load(['ValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsStandardized;

ValSignalsNormalized3 = load(['ValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsNormalized;
ValSignalsStandardized3 = load(['ValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsStandardized;

ValSignalsNormalized4 = load(['ValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsNormalized;
ValSignalsStandardized4 = load(['ValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllValSignalsStandardized;

AllValSignalsNormalized = cat(1, ValSignalsNormalized1, ValSignalsNormalized2, ValSignalsNormalized3, ValSignalsNormalized4);
AllValSignalsStandardized = cat(1, ValSignalsStandardized1, ValSignalsStandardized2, ValSignalsStandardized3, ValSignalsStandardized4);


save(['AllValSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValSignalsNormalized');
save(['AllValSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValSignalsStandardized');

%test
TestSignalsNormalized1 = load(['TestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsNormalized;
TestSignalsStandardized1 = load(['TestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

TestSignalsNormalized2 = load(['TestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsNormalized;
TestSignalsStandardized2 = load(['TestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

TestSignalsNormalized3 = load(['TestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsNormalized;
TestSignalsStandardized3 = load(['TestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

TestSignalsNormalized4 = load(['TestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsNormalized;
TestSignalsStandardized4 = load(['TestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).AllTestSignalsStandardized;

AllTestSignalsNormalized = cat(1, TestSignalsNormalized1, TestSignalsNormalized2, TestSignalsNormalized3, TestSignalsNormalized4);
AllTestSignalsStandardized = cat(1, TestSignalsStandardized1, TestSignalsStandardized2, TestSignalsStandardized3, TestSignalsStandardized4);


save(['AllTestSignalsNormalized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsNormalized');
save(['AllTestSignalsStandardized_', Tool, '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSignalsStandardized');

%% Concatenate Labels
TrainLabels1 = load(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
TrainLabels2 = load(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
TrainLabels3 = load(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
TrainLabels4 = load(['TrainLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainLabels;
AllTrainLabels = cat(1, TrainLabels1, TrainLabels2, TrainLabels3, TrainLabels4);
save(['AllTrainLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainLabels');

ValLabels1 = load(['ValLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValLabels;
ValLabels2 = load(['ValLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValLabels;
ValLabels3 = load(['ValLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValLabels;
ValLabels4 = load(['ValLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValLabels;
AllValLabels = cat(1, ValLabels1, ValLabels2, ValLabels3, ValLabels4);
save(['AllValLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValLabels');

TestLabels1 = load(['TestLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
TestLabels2 = load(['TestLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
TestLabels3 = load(['TestLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
TestLabels4 = load(['TestLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestLabels;
AllTestLabels = cat(1, TestLabels1, TestLabels2, TestLabels3, TestLabels4);
save(['AllTestLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestLabels');

%% Concatenate Segment Labels
TrainSegmentLabels1 = load(['TrainSegmentLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainSegmentLabels;
TrainSegmentLabels2 = load(['TrainSegmentLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainSegmentLabels;
TrainSegmentLabels3 = load(['TrainSegmentLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainSegmentLabels;
TrainSegmentLabels4 = load(['TrainSegmentLabelsElimOverlapNoMix_', num2str(max_overlap*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TrainSegmentLabels;
AllTrainSegmentLabels = cat(1, TrainSegmentLabels1, TrainSegmentLabels2, TrainSegmentLabels3, TrainSegmentLabels4);
save(['AllTrainSegmentLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTrainSegmentLabels');

ValSegmentLabels1 = load(['ValSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValSegmentLabels;
ValSegmentLabels2 = load(['ValSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValSegmentLabels;
ValSegmentLabels3 = load(['ValSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValSegmentLabels;
ValSegmentLabels4 = load(['ValSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).ValSegmentLabels;
AllValSegmentLabels = cat(1, ValSegmentLabels1, ValSegmentLabels2, ValSegmentLabels3, ValSegmentLabels4);
save(['AllValSegmentLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllValSegmentLabels');

TestSegmentLabels1 = load(['TestSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestSegmentLabels;
TestSegmentLabels2 = load(['TestSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestSegmentLabels;
TestSegmentLabels3 = load(['TestSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_vertical_left_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestSegmentLabels;
TestSegmentLabels4 = load(['TestSegmentLabelsElimOverlapNoMix_', num2str(0*100), '_horizontal_right_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TestSegmentLabels;
AllTestSegmentLabels = cat(1, TestSegmentLabels1, TestSegmentLabels2, TestSegmentLabels3, TestSegmentLabels4);
save(['AllTestSegmentLabels', '_', num2str(max_overlap*100), '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'AllTestSegmentLabels');

%% Dataset sizes and percentages
TrainLabels = TrainLabels4;
ValLabels = ValLabels4;
TestLabels = TestLabels4;

TrErr = length(find(TrainLabels == 1));
TrNot = length(find(TrainLabels == 0));

VErr = length(find(ValLabels == 1));
VNot = length(find(ValLabels == 0));

TeErr = length(find(TestLabels == 1));
TeNot = length(find(TestLabels == 0));

Err = TrErr + VErr + TeErr;
Not = TrNot + VNot + TeNot;

TrErr/Err
TrNot/Not

VErr/Err
VNot/Not

TeErr/Err
TeNot/Not