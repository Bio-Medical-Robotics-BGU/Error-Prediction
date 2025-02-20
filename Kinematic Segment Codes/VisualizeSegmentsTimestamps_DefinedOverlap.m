%% This code takes the labeled detected movements and the timesamples of the generated kinematic segments and
%visualizes them to show one of the kinematic features not before and
%before an error.
%This is Fig. 2

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";


MovesPath = fullfile(project_path, "AllMoves");
MatPath = fullfile(project_path, "AllMatFiles");
SegPath = fullfile(project_path, "AllSegmentations");
DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
KinematicsPath = fullfile(project_path, "SynchronizedKinematics50Hz");


participant = 'M';
month = '2';
tt = 'c';

%% Defining variables
seg_length = 50; %what segment length to take
max_overlap = 0; %50% max overlap
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 100 hz)

%%

cd(SegPath)
segmentation = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer_Segmentation.mat']).Segmentation;

% frame numbers of relevant part of video
start_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Right ring caught for the first time')));
end_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Right ring outside the tower')));
tower = 'vertical_right';
start_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Right ring caught for the first time')));
end_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Right ring outside the tower')));



tf = contains(tower, 'right');

if tf
    inds = 1:20;
else
    inds = 21:40;
end

tf2 = contains(tower, 'vertical');

if tf2
    ind = inds(1); %position in z axis
else
    ind = inds(3); %position in x axis
end


%loading moves and segments
cd(MovesPath)
Moves = load(['MovesFixed_', participant, '_', month, '_', tt, '_', tower, '.mat']).Moves;


cd(DataPath)
MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;
KinematicSegmentSamps = load(['KinematicSegmentSampsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).KinematicSegmentSamps;
TowerKinematics = load(['KinematicsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TowerKinematics;
kinematic_sig = TowerKinematics(:, :, ind);

cd(MatPath)
VidTimeStamps = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer.mat']).D.VideoTimeStamps;
TowerVidStamps = VidTimeStamps(start_ind:end_ind);

cd(KinematicsPath)
kinematics = load([participant, '_', month, '_', tt, '_', 'RingTowertransfer.mat']).T;
kinematicStamps = table2array(kinematics(:, 1));

[L, NumMoves] = bwlabel(Moves);

fig = figure;
grays = colormap(gray(length(find(MoveLabel == 0)) + 4));
grays = grays(1:end-4, :);
close(fig)

figure
hold on
% plot(TowerVidStamps, 0.05*Moves, 'color', [219 204 240]./255)
for j = 1 : NumMoves
    move_inds = find(L == j);
    patch('XData', [TowerVidStamps(move_inds(1)), TowerVidStamps(move_inds(end)), TowerVidStamps(move_inds(end)), TowerVidStamps(move_inds(1))], 'YData', [0, 0, 0.05, 0.05], 'facecolor', [233 224 246]./255, 'edgecolor', [233 224 246]./255);
end

c1 = 1;
for k = 1 : length(MoveLabel)
    if MoveLabel(k) == 0
        c = grays(c1, :);
        if c1 == 1
            c1 = size(grays, 1);
        else
            c1 = 1;
        end
    else
        c = 'r';
    end
    plot(linspace(kinematicStamps(KinematicSegmentSamps(k, 1)),kinematicStamps(KinematicSegmentSamps(k, 2)), seg_length), kinematic_sig(k, :), 'color', c, 'linewidth', 2)

end
set(gca, fontsize = 12, fontname = 'Times New Roman')

xlabel('Time [sec]', fontsize = 14, fontname = 'Times New Roman')
ylabel('Position X Axis [m]', fontsize = 14, fontname = 'Times New Roman')

% xticks = TowerVidStamps - TowerVidStamps(1);
% xticks2 = xticks(1:20:end);
% xticks2 = round(xticks2, 2);
% set(gca, 'XTick', TowerVidStamps(1:20:end), 'XTickLabel', xticks2)

xticks = kinematicStamps(KinematicSegmentSamps(1, 1)):1:kinematicStamps(KinematicSegmentSamps(end, 2));
xticks2 = xticks - kinematicStamps(KinematicSegmentSamps(1, 1));
xticks2 = round(xticks2, 2);
set(gca, 'XTick', xticks(1:2:end), 'XTickLabel', xticks2(1:2:end))
set(gca, 'YTick', 0:0.01:0.05)

% l = legend();
xlim([kinematicStamps(KinematicSegmentSamps(1, 1)), kinematicStamps(KinematicSegmentSamps(end, 2))])

