%% This code creates the train, validation and test sets
project_path = "C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction";

DataPath = fullfile(project_path, "KinematicSegmentsAndLabels50Hz");
SavePath = fullfile(project_path, "DatasetsTrainValTest_OneSplit");

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U'];

AllMonths = ['1', '2', '3', '4', '5', '6'];
AllTimes = ['a', 'b', 'c'];
towers = [{'vertical_right'}; {'horizontal_left'}; {'vertical_left'}; {'horizontal_right'}];
%% Defining variables

tower = towers{1};

seg_length = 50; %what segment length to take
max_overlap = 0; %50% max overlap
advance = 1; %how much in advance do we want to predict the error (in samples, data sampled at 100 hz)

train_size = 0.6;
val_size = 0.2;
test_size = 0.2;

%% Summary Tables
vnames = {'Participant', 'Month', 'Session', 'Tower', 'Num Not Before Error', 'Num Before Error'};

Emptys = {};
%% Run over all towers per participant
for pp = 1:length(AllParticipants)
    try
    participant = AllParticipants(pp);
    disp(participant)

    v1 = cell2table(repmat({participant}, 18, 1));
    v1.Properties.VariableNames = {'Participant'};

    v2 = array2table(repelem(1:6, 3*ones(6,1))');
    v2.Properties.VariableNames = {'Month'};

    v3 = cell2table(repmat([{'a'}, {'b'}, {'c'}]', 6, 1));
    v3.Properties.VariableNames = {'Session'};

    v4 = cell2table(repmat({tower}, 18, 1));
    v4.Properties.VariableNames = {'Tower'};

    v5 = array2table(zeros(18, 1));
    v5.Properties.VariableNames = {'Num Not Before Error'};

    v6 = array2table(zeros(18, 1));
    v6.Properties.VariableNames = {'Num Before Error'};

    ParticipantTable = [v1, v2, v3, v4, v5, v6];
    
    TrainTable = array2table(zeros(0,length(vnames)), 'VariableNames',vnames);
    ValTable = array2table(zeros(0,length(vnames)), 'VariableNames',vnames);
    TestTable = array2table(zeros(0,length(vnames)), 'VariableNames',vnames);
    
    

    for mm = 1:length(AllMonths)
        month = AllMonths(mm);
        disp(month)

        for ttt = 1:length(AllTimes)
            tt = AllTimes(ttt);
            disp(tt)

            try

                cd(DataPath)
                MoveLabel = load(['MoveLabelElimOverlapKin_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;

                ind1 = find(ParticipantTable.Month == str2num(month));
                ind2 = find(strcmp(ParticipantTable.Session, tt));
                ind = intersect(ind1, ind2); %this is the index in the table

                ParticipantTable(ind, 5) = array2table(length(find(MoveLabel == 0)));
                ParticipantTable(ind, 6) = array2table(length(find(MoveLabel == 1)));
                assert(length(find(MoveLabel == 0)) + length(find(MoveLabel == 1)) == length(MoveLabel))

            catch
                warning(['Problem in ', participant, ' ', month, ' ', tt, ' ', tower])
            end
        end
    end

    %take out rows that are zero in both types of segments
    rem = find((table2array(ParticipantTable(:, 6)) + table2array(ParticipantTable(:, 5))) == 0);
    ParticipantTable(rem, :) = [];

    %BEFORE ERROR SEGMENTS
    Total_err = sum(table2array(ParticipantTable(:, 6)));
    %all possible combinations of sums
    Op_Mat_err = (dec2bin(1:2^height(ParticipantTable)-1) - '0');
    allsums_err = (Op_Mat_err*table2array(ParticipantTable(:, 6)));
    assert(Total_err == max(allsums_err))

    %the set sizes we need for this participant's current tower
    curr_test_size_err = test_size.*Total_err;
    curr_val_size_err = val_size.*Total_err;
    curr_train_size_err = train_size.*Total_err;

    %find best way to split the data into groups
    %all the possible ways to get to each set
    min_test_err = find(abs(allsums_err - curr_test_size_err) == min(abs(allsums_err - curr_test_size_err)));
    min_val_err = find(abs(allsums_err - curr_val_size_err) == min(abs(allsums_err - curr_val_size_err)));
    min_train_err = find(abs(allsums_err - curr_train_size_err) == min(abs(allsums_err - curr_train_size_err)));

    %now we need to find the options that dont involve using the same
    %trials in different sets. This means, which way only has 1 in the
    %total of columns on the indices of Op_Mat of the three above indices
    test_options_err = Op_Mat_err(min_test_err, :);
    val_options_err = Op_Mat_err(min_val_err, :);
    train_options_err = Op_Mat_err(min_train_err, :);

    Possibles_err = [];

    for te = 1:length(min_test_err)
        test_cand = test_options_err(te, :);
        assert(all(test_cand == Op_Mat_err(min_test_err(te), :)))

        for va = 1:length(min_val_err)
            val_cand = val_options_err(va, :);
            assert(all(val_cand == Op_Mat_err(min_val_err(va), :)))

            for tr = 1:length(min_train_err)
                train_cand = train_options_err(tr, :);
                assert(all(train_cand == Op_Mat_err(min_train_err(tr), :)))

                summy = test_cand + val_cand + train_cand;
                if ~all(summy == 1)
                    continue
                else
                    Possibles_err = [Possibles_err; [min_test_err(te), min_val_err(va), min_train_err(tr)]];
                end
            end
        end
    end

    %If couldnt create the exact desired sizes:
    dist_err = 1; 
    while (isempty(Possibles_err) && dist_err <= 10) 

        %find best way to split the data into groups
        %all the possible ways to get to each set
        min_test_err = find(abs(allsums_err - curr_test_size_err) < dist_err);
        min_val_err = find(abs(allsums_err - curr_val_size_err) < dist_err);
        min_train_err = find(abs(allsums_err - curr_train_size_err) < dist_err);

        %now we need to find the options that dont involve using the same
        %trials in different sets. This means, which way only has 1 in the
        %total of columns on the indices of Op_Mat of the three above indices
        test_options_err = Op_Mat_err(min_test_err, :);
        val_options_err = Op_Mat_err(min_val_err, :);
        train_options_err = Op_Mat_err(min_train_err, :);

        Possibles_err = [];

        for te = 1:length(min_test_err)
            test_cand = test_options_err(te, :);
            assert(all(test_cand == Op_Mat_err(min_test_err(te), :)))

            for va = 1:length(min_val_err)
                val_cand = val_options_err(va, :);
                assert(all(val_cand == Op_Mat_err(min_val_err(va), :)))

                for tr = 1:length(min_train_err)
                    train_cand = train_options_err(tr, :);
                    assert(all(train_cand == Op_Mat_err(min_train_err(tr), :)))

                    summy = test_cand + val_cand + train_cand;
                    if ~all(summy == 1)
                        continue
                    else
                        Possibles_err = [Possibles_err; [min_test_err(te), min_val_err(va), min_train_err(tr)]];
                    end
                end
            end
        end
        dist_err = dist_err + 1;
    end

    %find the options that use all the trials:

    Possibles_err_all = Possibles_err;


    %NOT BEFORE ERROR SEGMENTS
    Total_not = sum(table2array(ParticipantTable(:, 5)));
    %all possible combinations of sums
    Op_Mat_not = (dec2bin(1:2^height(ParticipantTable)-1) - '0');
    allsums_not = (Op_Mat_not*table2array(ParticipantTable(:, 5)));
    assert(Total_not == max(allsums_not))

    %the set sizes we need for this participant's current tower
    curr_test_size_not = test_size.*Total_not;
    curr_val_size_not = val_size.*Total_not;
    curr_train_size_not = train_size.*Total_not;

    %find best way to split the data into groups
    %all the possible ways to get to each set
    min_test_not = find(abs(allsums_not - curr_test_size_not) == min(abs(allsums_not - curr_test_size_not)));
    min_val_not = find(abs(allsums_not - curr_val_size_not) == min(abs(allsums_not - curr_val_size_not)));
    min_train_not = find(abs(allsums_not - curr_train_size_not) == min(abs(allsums_not - curr_train_size_not)));

    %now we need to find the options that dont involve using the same
    %trials in different sets. This means, which way only has 1 in the
    %total of columns on the indices of Op_Mat of the three above indices
    test_options_not = Op_Mat_not(min_test_not, :);
    val_options_not = Op_Mat_not(min_val_not, :);
    train_options_not = Op_Mat_not(min_train_not, :);

    Possibles_not = [];

    for te = 1:length(min_test_not)
        test_cand = test_options_not(te, :);
        assert(all(test_cand == Op_Mat_not(min_test_not(te), :)))

        for va = 1:length(min_val_not)
            val_cand = val_options_not(va, :);
            assert(all(val_cand == Op_Mat_not(min_val_not(va), :)))

            for tr = 1:length(min_train_not)
                train_cand = train_options_not(tr, :);
                assert(all(train_cand == Op_Mat_not(min_train_not(tr), :)))

                summy = test_cand + val_cand + train_cand;
                if ~all(summy == 1)
                    continue
                else
                    Possibles_not = [Possibles_not; [min_test_not(te), min_val_not(va), min_train_not(tr)]];
                end
            end
        end
    end



    dist_not = 1;
    while (isempty(Possibles_not)  && dist_not <= 10)
        %find best way to split the data into groups
        %all the possible ways to get to each set
        min_test_not = find(abs(allsums_not - curr_test_size_not) < dist_not);
        min_val_not = find(abs(allsums_not - curr_val_size_not) < dist_not);
        min_train_not = find(abs(allsums_not - curr_train_size_not) < dist_not);

        %now we need to find the options that dont involve using the same
        %trials in different sets. This means, which way only has 1 in the
        %total of columns on the indices of Op_Mat of the three above indices
        test_options_not = Op_Mat_not(min_test_not, :);
        val_options_not = Op_Mat_not(min_val_not, :);
        train_options_not = Op_Mat_not(min_train_not, :);

        Possibles_not = [];

        for te = 1:length(min_test_not)
            test_cand = test_options_not(te, :);
            assert(all(test_cand == Op_Mat_not(min_test_not(te), :)))

            for va = 1:length(min_val_not)
                val_cand = val_options_not(va, :);
                assert(all(val_cand == Op_Mat_not(min_val_not(va), :)))

                for tr = 1:length(min_train_not)
                    train_cand = train_options_not(tr, :);
                    assert(all(train_cand == Op_Mat_not(min_train_not(tr), :)))

                    summy = test_cand + val_cand + train_cand;
                    if ~all(summy == 1)
                        continue
                    else
                        Possibles_not = [Possibles_not; [min_test_not(te), min_val_not(va), min_train_not(tr)]];
                    end
                end
            end
        end

        dist_not = dist_not + 1;

    end

    %find the options that use all the trials:
    Possibles_not_all = Possibles_not;


    if isempty (Possibles_not_all) || isempty(Possibles_err_all)
        warning(['Empty all in ', participant, ' ', tower])
        Emptys = [Emptys; participant]
        continue

    else
        [AllPossibles, n, e] = intersect(Possibles_not_all, Possibles_err_all, 'rows');
        if ~isempty(AllPossibles)
            chosen = AllPossibles(randi(size(AllPossibles, 1)), :);
        else
            %then we would like to find the option that is closest by
            %minimizing an error
            
            %we need to run over all the options
            
            %How big is the error for the not before segments if we use the
            %before error segments options?
            
            AllErrErrs = zeros(size(Possibles_err_all, 1), 1);
            
            for errs = 1 : size(Possibles_err_all, 1)
                test = find(Op_Mat_not(Possibles_err_all(errs, 1), :));
                val = find(Op_Mat_not(Possibles_err_all(errs, 2), :));
                train = find(Op_Mat_not(Possibles_err_all(errs, 3), :));
                
                test_sum = sum(table2array(    ParticipantTable(test, "Num Not Before Error")    ));
                val_sum = sum(table2array(    ParticipantTable(val, "Num Not Before Error")    ));
                train_sum = sum(table2array(    ParticipantTable(train, "Num Not Before Error")    ));
                
                test_error = abs(curr_test_size_not - test_sum);
                val_error = abs(curr_val_size_not - val_sum);
                train_error = abs(curr_train_size_not - train_sum);
                
                AllErrErrs(errs) = test_error + val_error + train_error;
                
                %check
                test_sum2 = sum(table2array(    ParticipantTable(test, "Num Before Error")    ));
                val_sum2 = sum(table2array(    ParticipantTable(val, "Num Before Error")    ));
                train_sum2 = sum(table2array(    ParticipantTable(train, "Num Before Error")    ));
                
                test_error2 = abs(curr_test_size_err - test_sum2);
                val_error2 = abs(curr_val_size_err - val_sum2);
                train_error2 = abs(curr_train_size_err - train_sum2);
                
                if dist_err == 1
                    assert ( test_error2 == min(abs(allsums_err - curr_test_size_err)))
                    assert ( val_error2 == min(abs(allsums_err - curr_val_size_err)))
                    assert ( train_error2 == min(abs(allsums_err - curr_train_size_err)))
                end
                
            end
            
            %How big is the error for the before error segments if we use the
            %not before error segments options?
            
            AllNotErrs = zeros(size(Possibles_not_all, 1) , 1);
            
            for nots = 1 : size(Possibles_not_all, 1) 
                test = find(Op_Mat_err(Possibles_not_all(nots, 1), :));
                val = find(Op_Mat_err(Possibles_not_all(nots, 2), :));
                train = find(Op_Mat_err(Possibles_not_all(nots, 3), :));
                
                test_sum = sum(table2array(    ParticipantTable(test, "Num Before Error")    ));
                val_sum = sum(table2array(    ParticipantTable(val, "Num Before Error")    ));
                train_sum = sum(table2array(    ParticipantTable(train, "Num Before Error")    ));
                
                test_error = abs(curr_test_size_err - test_sum);
                val_error = abs(curr_val_size_err - val_sum);
                train_error = abs(curr_train_size_err - train_sum);
                
                AllNotErrs(nots) = test_error + val_error + train_error;
                
                %check
                test_sum2 = sum(table2array(    ParticipantTable(test, "Num Not Before Error")    ));
                val_sum2 = sum(table2array(    ParticipantTable(val, "Num Not Before Error")    ));
                train_sum2 = sum(table2array(    ParticipantTable(train, "Num Not Before Error")    ));
                
                test_error2 = abs(curr_test_size_not - test_sum2);
                val_error2 = abs(curr_val_size_not - val_sum2);
                train_error2 = abs(curr_train_size_not - train_sum2);
                
                if dist_not == 1
                    assert ( test_error2 == min(abs(allsums_not - curr_test_size_not)))
                    assert ( val_error2 == min(abs(allsums_not - curr_val_size_not)))
                    assert ( train_error2 == min(abs(allsums_not - curr_train_size_not)))
                end
                
            end
            
            MinErr = min(AllErrErrs);
            MinNot = min(AllNotErrs);
            
            %use the option that gives the smallest error
            if MinErr <= MinNot %then use one of the best options from the options that give optimal 
                %results for the before error segments
                
                %all options that give the minimum error
                all_options = find(AllErrErrs == MinErr);
                chosen_ind =  all_options(randi(size(all_options, 1)));
                chosen = Possibles_err_all(chosen_ind, :);
                
                %check
                test = find(Op_Mat_not(chosen(1), :));
                val = find(Op_Mat_not(chosen(2), :));
                train = find(Op_Mat_not(chosen(3), :));
                
                test_sum = sum(table2array(    ParticipantTable(test, "Num Not Before Error")    ));
                val_sum = sum(table2array(    ParticipantTable(val, "Num Not Before Error")    ));
                train_sum = sum(table2array(    ParticipantTable(train, "Num Not Before Error")    ));
                
                test_error = abs(curr_test_size_not - test_sum);
                val_error = abs(curr_val_size_not - val_sum);
                train_error = abs(curr_train_size_not - train_sum);
                
                TotalError = test_error + val_error + train_error;
                
                assert(TotalError == MinErr)
                
            else %then use one of the best options from the options that give optimal 
                %results for the not before error segments
                
                %all options that give the minimum error
                all_options = find(AllNotErrs == MinNot);
                chosen_ind =  all_options(randi(size(all_options, 1)));
                chosen = Possibles_not_all(chosen_ind, :);
                
                %check
                test = find(Op_Mat_err(chosen(1), :));
                val = find(Op_Mat_err(chosen(2), :));
                train = find(Op_Mat_err(chosen(3), :));
                
                test_sum = sum(table2array(    ParticipantTable(test, "Num Before Error")    ));
                val_sum = sum(table2array(    ParticipantTable(val, "Num Before Error")    ));
                train_sum = sum(table2array(    ParticipantTable(train, "Num Before Error")    ));
                
                test_error = abs(curr_test_size_err - test_sum);
                val_error = abs(curr_val_size_err - val_sum);
                train_error = abs(curr_train_size_err - train_sum);
                
                TotalError = test_error + val_error + train_error;
                
                assert(TotalError == MinNot)

            end

            
        end

    end

    %fill in tables
    final_test = find(Op_Mat_err(chosen(1), :));
    final_val = find(Op_Mat_err(chosen(2), :));
    final_train = find(Op_Mat_err(chosen(3), :));


    final_test2 = find(Op_Mat_not(chosen(1), :));
    final_val2 = find(Op_Mat_not(chosen(2), :));
    final_train2 = find(Op_Mat_not(chosen(3), :));

    assert (all(final_test == final_test2))
    assert (all(final_val == final_val2))
    assert (all(final_train == final_train2))


    for i = 1:length(final_test)
        TestTable = [TestTable; {{0}, 0, {0}, {0}, 0, 0}];

        TestTable(height(TestTable), 1) = {participant};
        TestTable(height(TestTable), 2) = ParticipantTable(final_test(i), 2);
        TestTable(height(TestTable), 3) = ParticipantTable(final_test(i), 3);
        TestTable(height(TestTable), 4) = ParticipantTable(final_test(i), 4);
        TestTable(height(TestTable), 5) = ParticipantTable(final_test(i), 5);
        TestTable(height(TestTable), 6) = ParticipantTable(final_test(i), 6);

    end

    for i = 1:length(final_val)
        ValTable = [ValTable; {{0}, 0, {0}, {0}, 0, 0}];

        ValTable(height(ValTable), 1) = {participant};
        ValTable(height(ValTable), 2) = ParticipantTable(final_val(i), 2);
        ValTable(height(ValTable), 3) = ParticipantTable(final_val(i), 3);
        ValTable(height(ValTable), 4) = ParticipantTable(final_val(i), 4);
        ValTable(height(ValTable), 5) = ParticipantTable(final_val(i), 5);
        ValTable(height(ValTable), 6) = ParticipantTable(final_val(i), 6);

    end

    for i = 1:length(final_train)
        TrainTable = [TrainTable; {{0}, 0, {0}, {0}, 0, 0}];

        TrainTable(height(TrainTable), 1) = {participant};
        TrainTable(height(TrainTable), 2) = ParticipantTable(final_train(i), 2);
        TrainTable(height(TrainTable), 3) = ParticipantTable(final_train(i), 3);
        TrainTable(height(TrainTable), 4) = ParticipantTable(final_train(i), 4);
        TrainTable(height(TrainTable), 5) = ParticipantTable(final_train(i), 5);
        TrainTable(height(TrainTable), 6) = ParticipantTable(final_train(i), 6);

    end

    cd(SavePath)
%     save(['TrainTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], 'TrainTable')
%     save(['ValTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], 'ValTable')
%     save(['TestTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], 'TestTable')

    parsave(['TrainTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], TrainTable)
    parsave(['ValTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], ValTable)
    parsave(['TestTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], TestTable)

    catch
        warning(['Error in participant ' participant])
    end

end

%% Checking
Emptys = [];

AllGood = setdiff(AllParticipants, Emptys); %all the participants for who the algorithm managed to find a split
%first load them and count how many of each kine of segment we have in each
%of the three datasets

for pp = 1:length(AllGood)
    participant = AllGood(pp);
    disp(participant)
    
    cd(SavePath)
    try
        TrainTable = load(['TrainTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz.mat']).data;
    catch
        TrainTable = [];
    end
    try
        ValTable = load(['ValTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz.mat']).data;
    catch
        ValTable = [];
    end
    try
        TestTable = load(['TestTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz.mat']).data;
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

end %end of running over good participants

%check dataset sizes
TotalTrainErr = sum(table2array(AllTrains(:, "Num Before Error")));
TotalTrainNot = sum(table2array(AllTrains(:, "Num Not Before Error")));

TotalValErr = sum(table2array(AllVals(:, "Num Before Error")));
TotalValNot = sum(table2array(AllVals(:, "Num Not Before Error")));

TotalTestErr = sum(table2array(AllTests(:, "Num Before Error")));
TotalTestNot = sum(table2array(AllTests(:, "Num Not Before Error")));

TotalErr = TotalTrainErr + TotalValErr + TotalTestErr;
TotalNot = TotalTrainNot + TotalValNot + TotalTestNot;
    
(TotalTrainNot / TotalNot )*100
(TotalValNot / TotalNot )*100
(TotalTestNot / TotalNot )*100

(TotalTrainErr / TotalErr )*100
(TotalValErr / TotalErr )*100
(TotalTestErr / TotalErr )*100

% run over the emptys in random order
inds = randperm(length(Emptys));

for pp = 1:length(Emptys)
    participant = Emptys(inds(pp));
    
    disp(participant)

    v1 = cell2table(repmat({participant}, 18, 1));
    v1.Properties.VariableNames = {'Participant'};

    v2 = array2table(repelem(1:6, 3*ones(6,1))');
    v2.Properties.VariableNames = {'Month'};

    v3 = cell2table(repmat([{'a'}, {'b'}, {'c'}]', 6, 1));
    v3.Properties.VariableNames = {'Session'};

    v4 = cell2table(repmat({tower}, 18, 1));
    v4.Properties.VariableNames = {'Tower'};

    v5 = array2table(zeros(18, 1));
    v5.Properties.VariableNames = {'Num Not Before Error'};

    v6 = array2table(zeros(18, 1));
    v6.Properties.VariableNames = {'Num Before Error'};

    ParticipantTable = [v1, v2, v3, v4, v5, v6];
    
    
    for mm = 1:length(AllMonths)
        month = AllMonths(mm);
        disp(month)

        for ttt = 1:length(AllTimes)
            tt = AllTimes(ttt);
            disp(tt)

            try

                cd(DataPath)
                MoveLabel = load(['MoveLabelElimOverlap_', num2str(max_overlap*100), '_', participant, '_', month, '_', tt, '_', tower, '_len', num2str(seg_length), 'ad_', num2str(advance), '_50Hz', '.mat']).MoveLabel;

                ind1 = find(ParticipantTable.Month == str2num(month));
                ind2 = find(strcmp(ParticipantTable.Session, tt));
                ind = intersect(ind1, ind2); %this is the index in the table

                ParticipantTable(ind, 5) = array2table(length(find(MoveLabel == 0)));
                ParticipantTable(ind, 6) = array2table(length(find(MoveLabel == 1)));
                assert(length(find(MoveLabel == 0)) + length(find(MoveLabel == 1)) == length(MoveLabel))

            catch
                warning(['Problem in ', participant, ' ', month, ' ', tt, ' ', tower])
            end
        end
    end

    %take out rows that are zero in both types of segments
    rem = find((table2array(ParticipantTable(:, 6)) + table2array(ParticipantTable(:, 5))) == 0);
    ParticipantTable(rem, :) = [];
    
    %the number of segments in each class for this participant
    Participant_err = sum(table2array(ParticipantTable(:, 6)));
    Participant_not = sum(table2array(ParticipantTable(:, 5)));
    
    %generally, the lack of ability to find a possible division stemmed
    %from a very small number of segments.
    %Therefore, we will add all of them to one of the three datasets, based
    %on which would lead to the smallest deviation of that dataset from the
    %desired size
    
    %the current number of segments in each dataset from each class
    TotalTrainErr = sum(table2array(AllTrains(:, "Num Before Error")));
    TotalTrainNot = sum(table2array(AllTrains(:, "Num Not Before Error")));
    
    TotalValErr = sum(table2array(AllVals(:, "Num Before Error")));
    TotalValNot = sum(table2array(AllVals(:, "Num Not Before Error")));
    
    TotalTestErr = sum(table2array(AllTests(:, "Num Before Error")));
    TotalTestNot = sum(table2array(AllTests(:, "Num Not Before Error")));
    
    TotalErr = TotalTrainErr + TotalValErr + TotalTestErr;
    TotalNot = TotalTrainNot + TotalValNot + TotalTestNot;
    
    NewTrainDiff = abs((TotalTrainErr + Participant_err) - 0.6*(TotalErr + Participant_err)) + abs((TotalTrainNot + Participant_not) - 0.6*(TotalNot + Participant_not));
    CurrTrainDiff = abs(TotalTrainErr - 0.6*TotalErr) + abs(TotalTrainNot - 0.6*TotalNot);
    TrainDiff = CurrTrainDiff - NewTrainDiff;
    %the difference between the real size of the dataset and that that we
    %should have if we add all the segments to the train class.
    
    %the same for the validation and test sets:
    NewValDiff = abs((TotalValErr + Participant_err) - 0.2*(TotalErr + Participant_err)) + abs((TotalValNot + Participant_not) - 0.2*(TotalNot + Participant_not));
    CurrValDiff = abs(TotalValErr - 0.2*TotalErr) + abs(TotalValNot - 0.2*TotalNot);
    ValDiff = CurrValDiff - NewValDiff;
    
    NewTestDiff = abs((TotalTestErr + Participant_err) - 0.2*(TotalErr + Participant_err)) + abs((TotalTestNot + Participant_not) - 0.2*(TotalNot + Participant_not));
    CurrTestDiff = abs(TotalTestErr - 0.2*TotalErr) + abs(TotalTestNot - 0.2*TotalNot);
    TestDiff = CurrTestDiff - NewTestDiff;

    smallest = find([TrainDiff, ValDiff, TestDiff] == max([TrainDiff, ValDiff, TestDiff]));
    smallest = smallest(randi(length(smallest)));
    cd(SavePath)
    if smallest == 1 %add to train
        parsave(['TrainTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], ParticipantTable)
        AllTrains = [AllTrains; ParticipantTable];
    elseif smallest == 2 %add to validation
        parsave(['ValTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], ParticipantTable)
        AllVals = [AllVals; ParticipantTable];
    else %add to test
        parsave(['TestTable_', tower, participant, '_overlap', num2str(max_overlap*100), '_len', num2str(seg_length), '_ad', num2str(advance), '_50Hz'], ParticipantTable)
        AllTests = [AllTests; ParticipantTable];
    end
    
end %end of running over the empty participants

%check dataset sizes
TotalTrainErr = table2array(sum(AllTrains(:, "Num Before Error")));
TotalTrainNot = table2array(sum(AllTrains(:, "Num Not Before Error")));

TotalValErr = table2array(sum(AllVals(:, "Num Before Error")));
TotalValNot = table2array(sum(AllVals(:, "Num Not Before Error")));

TotalTestErr = table2array(sum(AllTests(:, "Num Before Error")));
TotalTestNot = table2array(sum(AllTests(:, "Num Not Before Error")));

TotalErr = TotalTrainErr + TotalValErr + TotalTestErr;
TotalNot = TotalTrainNot + TotalValNot + TotalTestNot;




