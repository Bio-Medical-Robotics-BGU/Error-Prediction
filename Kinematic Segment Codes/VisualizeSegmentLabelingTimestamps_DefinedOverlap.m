%% This code takes the labeled detected movements and the timesamples of the generated kinematic segments and
%visualizes them to ensure that the labeling of the kinematic segments was
%correct. The difference between this code and the other visualization code
%is that this code uses the timestamps for everything, and is therefore a
%better check

project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";


MovesPath = fullfile(project_path, "AllMoves");
MatPath = fullfile(project_path, "AllMatFiles");
SegPath = fullfile(project_path, "AllSegmentations");

DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
KinematicsPath = fullfile(project_path, "SynchronizedKinematics50Hz");

% participant = input('Which participant would you like? (A - U) \n', 's');
% month = input('Which month would you like? (1 - 6) \n', 's');
% tt = input('Which session would you like? (a - c) \n', 's');

% AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
AllParticipants = ['F'];
% AllMonths = ['1', '2', '3', '4', '5', '6'];
AllMonths = ['1'];
AllTimes = ['c'];


%% Defining variables
seg_length = 50; %what segment length to take
max_overlap = 0; %50% max overlap
advance = 25; %how much in advance do we want to predict the error (in samples, data sampled at 100 hz)

%%
for pp = 1:length(AllParticipants)
    participant = AllParticipants(pp);
    disp(participant)

    for mm = 1:length(AllMonths)
        month = AllMonths(mm);
        disp(month)

        for ttt = 1:length(AllTimes)
            tt = AllTimes(ttt);
            disp(tt)

            cd(SegPath)
            segmentation = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer_Segmentation.mat']).Segmentation;

            % 4. Run over the four towers
            for i = 1:4
                try

                    if i == 1 % vertical right
                        % frame numbers of relevant part of video
                        start_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Right ring caught for the first time')));
                        end_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Right ring outside the tower')));
                        tower = 'vertical_right';
                        start_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Right ring caught for the first time')));
                        end_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Right ring outside the tower')));

                    elseif i == 2  % horizontal left
                        % frame numbers of relevant part of video
                        start_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Start inserting ring to left bottom tower')));
                        end_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Ring placed - Bottom left tower')));
                        tower = 'horizontal_left';
                        start_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Start inserting ring to left bottom tower')));
                        end_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Ring placed - Bottom left tower')));

                    elseif i == 3  % vertical left
                        % frame numbers of relevant part of video
                        start_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Left ring caught for the first time')));
                        end_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Left ring outside the tower')));
                        tower = 'vertical_left';
                        start_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Left ring caught for the first time')));
                        end_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Left ring outside the tower')));

                    else % i=4, horizontal right
                        % frame numbers of relevant part of video
                        start_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Start inserting ring to right bottom tower')));
                        end_ind = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Finish Task')));
                        tower = 'horizontal_right';
                        start_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Start inserting ring to right bottom tower')));
                        end_stamp = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Finish Task')));

                    end %end of setting up tower parameters

                    if length(start_ind) > 1
                        start_ind = start_ind(1);
                        start_stamp = start_stamp(1);
                    end
                    if length(end_ind) > 1
                        end_ind = end_ind(end);
                        end_stamp = end_stamp(end);
                    end


                    %loading moves and segments
                    cd(MovesPath)
                    Moves = load(['MovesFixed_', participant, '_', month, '_', tt, '_', tower, '.mat']).Moves;


                    cd(DataPath)
                    MoveLabel = load(['MoveLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;
                    VideoSegmentSamps = load(['VideoSegmentSampsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).VideoSegmentSamps;
                    KinematicSegmentSamps = load(['KinematicSegmentSampsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).KinematicSegmentSamps;

                    cd(MatPath)
                    VidTimeStamps = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer.mat']).D.VideoTimeStamps;
                    TowerVidStamps = VidTimeStamps(start_ind:end_ind);

                    cd(KinematicsPath)
                    kinematics = load([participant, '_', month, '_', tt, '_', 'RingTowertransfer.mat']).T;
                    kinematicStamps = table2array(kinematics(:, 1));

                    [L, NumMoves] = bwlabel(Moves);

                    figure
                    hold on
                    plot(TowerVidStamps, Moves, 'color', [0.3 0.3 0.3])
                    for j = 1 : NumMoves
                        move_inds = find(L == j);
                        patch('XData', [TowerVidStamps(move_inds(1)), TowerVidStamps(move_inds(end)), TowerVidStamps(move_inds(end)), TowerVidStamps(move_inds(1))], 'YData', [0, 0, 1, 1], 'facecolor', [0.5 0.5 0.5])
                    end

                    y_vals = linspace(0.02, 1, length(MoveLabel));
                    for k = 1 : length(MoveLabel)
                        if MoveLabel(k) == 0
                            c = 'k';
                        else
                            c = 'r';
                        end
                        plot([kinematicStamps(KinematicSegmentSamps(k, 1)), kinematicStamps(KinematicSegmentSamps(k, 2))], [y_vals(k), y_vals(k)], 'color', c)

                    end
                    xlabel('Samples', fontsize = 14, fontname = 'Times New Roman')
                    set(gca, fontsize = 12, fontname = 'Times New Roman')

                    set(gca, 'Ytick', [])
                    title([participant, month, tt, ' ', tower], 'Interpreter', 'none', fontsize = 14, fontname = 'Times New Roman')
                catch
                    warning(['Problem in ', participant, ' ', month, ' ', tt, ' ', tower])
                end
            end
        end
    end
end

