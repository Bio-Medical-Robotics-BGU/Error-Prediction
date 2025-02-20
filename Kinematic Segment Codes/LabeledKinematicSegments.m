%% This code creates the kinematic segments and assigns two labels:
%MoveLabel - 1 if the segment is followed by an error (tower movement), 0
%if not.
%TowerSegmentLabel - where in the tower is the segment located.
project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";

SegPath = fullfile(project_path, "AllSegmentations");
MatPath = fullfile(project_path, "AllMatFiles");

MovesPath = fullfile(project_path, "AllMoves");
SegmentsPath = fullfile(project_path, "TowerSegments");
KinematicsPath = fullfile(project_path, "SynchronizedKinematics50Hz");

SavePath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
 
AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
AllMonths = ['1', '2', '3', '4', '5', '6'];
AllTimes = ['a', 'b', 'c'];

%% Defining variables
mult = 1;
seg_length = mult*50; %what segment length to take
overlap = seg_length - 1; %at this stage, there is nearly complete overlap between consecutive segments
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 50 hz)

%% Kinematic segments and two kinds of labels
cd(project_path)
diary('WarningLog')
for pp = 1:length(AllParticipants)
    participant = AllParticipants(pp);
    disp(participant)

    for mm = 1:length(AllMonths)
        month = AllMonths(mm);
        disp(month)

        for ttt = 1:length(AllTimes)
            tt = AllTimes(ttt);
            disp(tt)

            if (participant == 'D' && month == '3' && tt == 'b')
                continue
            end

            if (participant == 'N' && month == '5' && tt == 'a')
                continue
            end
            if (participant == 'N' && month == '5' && tt == 'b')
                continue
            end

            if (participant == 'N' && month == '6' && tt == 'a')
                continue
            end
            if (participant == 'N' && month == '6' && tt == 'b')
                continue
            end

            try
                % 1. loading segmentation
                cd(SegPath)
                segmentation = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer_Segmentation.mat']).Segmentation;

                cd(MatPath)
                VidTimeStamps = load(['subj_', participant, '_', month, '_', tt, '_Ringtowertransfer.mat']).D.VideoTimeStamps;
            catch
                warning(['Problem in ', participant, ' ', month, ' ', tt,])
                continue
            end

            if any(strcmp(segmentation.Event, 'Robot shut down')) %if there was a robot shut down
                ShutDowns = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Robot shut down')));
                Restarts = segmentation.VideoFrameNumber(find(strcmp(segmentation.Event, 'Back to task')));

                ShutDownStamps = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Robot shut down')));
                RestartStamps = segmentation.TimeStamp(find(strcmp(segmentation.Event, 'Back to task')));
            else
                if exist('ShutDowns','var')
                    clearvars ShutDowns Restarts ShutDownStamps RestartStamps
                end

            end

            %kinematics
            cd(KinematicsPath)
            kinematics = load([participant, '_', month, '_', tt, '_', 'RingTowertransfer.mat']).T;

            % 4. Run over the four towers
            for i = 1:4

                % Tower Parameters
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
                try
                    cd(MovesPath)
                    Moves = load(['MovesFixed_', participant, '_', month, '_', tt, '_', tower, '.mat']).Moves;

                    cd(SegmentsPath)
                    Segments = load(['TowerSegmentsFixed_', participant, '_', month, '_', tt, '_', tower, '.mat']).Segments;
                catch
                    warning(['Problem in ', participant, ' ', month, ' ', tt, '_', tower])

                    continue
                end
                %Tower data
             
                TowerKinematics = [];
                MoveLabel = [];
                TowerSegmentLabel = [];
                VideoSegmentSamps = [];
                KinematicSegmentSamps = [];

                %tower video timestamps
                TowerVidStamps = VidTimeStamps(start_ind:end_ind);
                assert(TowerVidStamps(1) == start_stamp)
                assert(TowerVidStamps(end) == end_stamp)


                %running over all the samples in the tower

                %finding the data in this tower
                [min1, start] = min(abs(table2array(kinematics(:, 1)) - start_stamp));
                [min2, finish] = min(abs(table2array(kinematics(:, 1)) - end_stamp));

                s = start;
                while s <= finish - seg_length - 10 %such that there are still at least 10 task samples left
                    %after the segment
                    samples = [s, s + seg_length - 1]; %take seg_length samples
                    %ensure that the sampling frequency in the
                    %synchronized data is constant such that each
                    %segment contains the same amount of data

                    % get kinematic segment of length seg_length
                    kin_seg = table2array(kinematics(samples(1):samples(2), 4:end));

                    kin_stamps = table2array(kinematics([samples(1),samples(2)], 1));


                    assert(diff(kin_stamps) - mult*0.99 < 10e-5) %needs to be 0.98 for seg length 50

                    %finding the closest video timestamps for kinematic
                    %segment
                    [min3, vidsamp1] = min(abs(TowerVidStamps - kin_stamps(1)));
                    [min4, vidsamp2] = min(abs(TowerVidStamps - kin_stamps(2)));

                    % advance video timestamp
                    advance_kin_sample = s + seg_length - 1 + advance - 1;
                    advance_kin_stamp = table2array(kinematics(advance_kin_sample, 1));
                    [min5, vidsamp_advance_lim] = min(abs(TowerVidStamps - advance_kin_stamp));
                    if advance == 1
                        assert(advance_kin_sample == samples(2))
                        assert(vidsamp_advance_lim == vidsamp2)
                    end

                    %label for moves
                    label_kin_sample = s + seg_length - 1 + advance; %this is the sample after the segment
                    label_kin_stamp = table2array(kinematics(label_kin_sample, 1));
                    [min6, vidsamp_label] = min(abs(TowerVidStamps - label_kin_stamp));

                    %checking if this segment contains a tower movement
                    %If so - discard
                    %otherwise - save
                    %note - due to the different sampling frequencies
                    %(30hz for video and 50hz or 100hz for kinematics), if the
                    %last video sample contains a movement and the
                    %label sample == the last video sample, we won't
                    %discard the sample. This is because there are
                    %two or three kinematic samples in each video one.

                    SegMoves = sum(Moves(vidsamp1:vidsamp_advance_lim));

                    %MAKE SURE THAT THIS SEGMENT DOES NOT CONTAIN ROBOT
                    %SHUT DOWN - it can't because those are all errors

                    if ((SegMoves == 0) ||  (SegMoves == 1 && sum(Moves(vidsamp1:vidsamp_advance_lim - 1)) == 0 && vidsamp_label == vidsamp_advance_lim))
                        %either there are no detected errors in the
                        %segment or in the advance samples, or:
                        %there are, but it's only the last sample, and
                        %the video label sample and the last sample of
                        %the segment are the same, such that the error
                        %could have started either end the very end of
                        %the segment or after if
                        if (vidsamp_label <= length(Moves)) %ensuring that there is a sample for the label
                            TowerKinematics = cat(1, TowerKinematics, reshape(kin_seg, 1, seg_length, size(kin_seg, 2)));
                            MoveLabel = [MoveLabel; Moves(vidsamp_label)]; %if 1 - segment followed by error
                            %othersize, segment not followed by error
                            towersegs = mean(Segments(vidsamp1:vidsamp2));
                            TowerSegmentLabel = [TowerSegmentLabel; towersegs]; %describes the segments the tower contains (1-4)
                            %for checking on video
                            VideoSegmentSamps = [VideoSegmentSamps; [vidsamp1, vidsamp2]];
                            %for defining overlap
                            KinematicSegmentSamps = [KinematicSegmentSamps; samples];
                            %if we took this segment -> jump forward based
                            %on overlap
                            s = s + seg_length - overlap;
                        end

                    else
                        %if we didnt take the segment -> move forward
                        %by one and check if next segment still is in
                        %the error
                        s = s + 1;
                    end



                end%end of running over this towers samples

                %save this tower's kinematics and two label vectors

                cd(SavePath)
                save(['Kinematics_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TowerKinematics')
                save(['MoveLabel_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'MoveLabel')
                save(['TowerSegmentLabel_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TowerSegmentLabel')
                save(['VideoSegmentSamps_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'VideoSegmentSamps')
                save(['KinematicSegmentSamps_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'KinematicSegmentSamps')

            end %end of running over 4 towers


        end %end of running on participant's 3 sessions

    end %end of running on participant's 6 months

end %end of running on all participants

diary off