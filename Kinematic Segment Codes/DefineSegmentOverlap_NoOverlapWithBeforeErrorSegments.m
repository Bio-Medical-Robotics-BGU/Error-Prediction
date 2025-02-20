%% the purpose of this code is to reduce the overlapping between kinematic segments to the desired ratio
%while eliminating the possibilty of overlap between the classes

%replace the following line with the directories to the saved videos and
%segmentation
project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";

MovesSavePath = fullfile(project_path, "AllMoves");

DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");

seg_length = 100;
advance = 1;

max_overlap = 0; 

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];
AllMonths = ['1', '2', '3', '4', '5', '6'];
AllTimes = ['a', 'b', 'c'];

towers = {'vertical_right', 'horizontal_left', 'vertical_left', 'horizontal_right'};
%%
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


            % Run over the four towers
            for i = 1:4
                tower = towers{i};

                try

                    %loading moves and segments
                    cd(DataPath)
                    TowerKinematics = load(['Kinematics_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TowerKinematics;
                    MoveLabel = load(['MoveLabel_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).MoveLabel;
                    TowerSegmentLabel = load(['TowerSegmentLabel_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).TowerSegmentLabel;
                    VideoSegmentSamps = load(['VideoSegmentSamps_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).VideoSegmentSamps;
                    KinematicSegmentSamps = load(['KinematicSegmentSamps_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz.mat']).KinematicSegmentSamps;
                    
                catch
                    warning(['Problem in ', participant, ' ', month, ' ', tt, ' ', tower])
                    continue
                end

                 if isempty(KinematicSegmentSamps)
                    warning(['Empty in ', participant, ' ', month, ' ', tt, ' ', tower])
                    continue
                 end

                    cur_seg = KinematicSegmentSamps(1, 1):KinematicSegmentSamps(1, 2);
                    cur_label = MoveLabel(1);
                    cur_s = 1;

                    del_inds = [];


                    s = cur_s;

                    while s < length(MoveLabel)
                  
                        next_seg = KinematicSegmentSamps(s + 1, 1):KinematicSegmentSamps(s + 1, 2);
                        assert (length(next_seg) == seg_length)
                        over = length(intersect(cur_seg, next_seg));
                        if over/length(cur_seg) > max_overlap
                            %cases:
                            %1. if both are not before error segments -
                            %stick with the current
                            if cur_label == 0 && MoveLabel(s + 1) == 0
                                del_inds = [del_inds; s + 1];
                                %cur_seg = cur_seg; cur_label = cur_label;
                                %cur_s = cur_s;
                                s = s + 1; %check the next segment

                            %2. current is not before error and next is - take next
                            elseif cur_label == 0 && MoveLabel(s + 1) == 1
                                del_inds = [del_inds; cur_s];
                                cur_seg = next_seg;
                                cur_label = 1;
                                cur_s = s + 1;
                                s = s + 1; %go to the segment after next

                            %3. both are before error 
                            elseif cur_label == 1 && MoveLabel(s + 1) == 1
                                % assert((over/length(cur_seg))  == 1)
                                del_inds = [del_inds; cur_s];
                                cur_seg = next_seg;
                                cur_s = s + 1;
                                s = s + 1; %go to the segment after next

                            %4. current is before error and next is not -
                            %in this case they are seperated by an error
                            %and there should be no overlap - this case
                            %should never be reached
                            elseif cur_label == 1 && MoveLabel(s + 1) == 0
                                error('reached condition that should not be reached')
                            else
                                error('no case was entered')
                            end

                        else %not too much overlap
                            cur_seg = next_seg;
                            cur_label = MoveLabel(s + 1);
                            cur_s = s + 1;
                            s = s + 1; %go to the segment after next
                            
                        end %end of overlap condition

                    end%end of running over original segments

                    TowerKinematics(del_inds, :, :) = [];
                    MoveLabel(del_inds) = [];
                    TowerSegmentLabel(del_inds) = [];
                    VideoSegmentSamps(del_inds, :) = [];
                    KinematicSegmentSamps(del_inds, :) = [];

                    %removing any overlap between the classes
                    AllErrs = find(MoveLabel == 1);
                    del_inds_e = [];
                    if (any(AllErrs == 1))
                        AllErrs(find(AllErrs == 1)) = [];
                    end
                    for e = 1:length(AllErrs) %for each error, delete all overlapping segments
                        %preceding it
                        over = 1;
                        f = 1;
                        while over ~= 0

                            cur_seg = KinematicSegmentSamps(AllErrs(e), 1):KinematicSegmentSamps(AllErrs(e), 2);
                            prev_seg = KinematicSegmentSamps(AllErrs(e) - f, 1):KinematicSegmentSamps(AllErrs(e) - f, 2);
                            over = length(intersect(cur_seg, prev_seg));
                            if over ~= 0
                                del_inds_e = [del_inds_e; AllErrs(e) - f];
                            end

                            f = f + 1;

                            if (AllErrs(e) - f == 0)
                                over = 0;
                            end
                        end

                    end

                    TowerKinematics(del_inds_e, :, :) = [];
                    MoveLabel(del_inds_e) = [];
                    TowerSegmentLabel(del_inds_e) = [];
                    VideoSegmentSamps(del_inds_e, :) = [];
                    KinematicSegmentSamps(del_inds_e, :) = [];

                    %removing the 1 segments from the horizontal towers
                    if (strcmp(tower, 'horizontal_left') || strcmp(tower, 'horizontal_right'))
                        del_ones = find(TowerSegmentLabel < 2); %meaning at least part of it falls in segment 1
                        disp(length(del_ones))

                        TowerKinematics(del_ones, :, :) = [];
                        MoveLabel(del_ones) = [];
                        TowerSegmentLabel(del_ones) = [];
                        VideoSegmentSamps(del_ones, :) = [];
                        KinematicSegmentSamps(del_ones, :) = [];


                    end
                  

                    cd(DataPath)
                    save(['KinematicsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TowerKinematics')
                    save(['MoveLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'MoveLabel')
                    save(['TowerSegmentLabelElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'TowerSegmentLabel')
                    save(['VideoSegmentSampsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'VideoSegmentSamps')
                    save(['KinematicSegmentSampsElimOverlapNoMix_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz'], 'KinematicSegmentSamps')
             

            end %end of running over 4 towers


        end %end of running on participant's 3 sessions

    end %end of running on participant's 6 months

end %end of running on all participants

diary off