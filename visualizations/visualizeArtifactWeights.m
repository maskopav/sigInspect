% Patient data
PatientID = {'sig_1', 'sig_10', 'sig_12', 'sig_13', 'sig_14', 'sig_15', ...
             'sig_16', 'sig_17', 'sig_18', 'sig_19', 'sig_2', 'sig_20', ...
             'sig_21', 'sig_22', 'sig_23', 'sig_24', 'sig_25', 'sig_26', ...
             'sig_27', 'sig_29', 'sig_3', 'sig_4', 'sig_5', 'sig_6', ...
             'sig_7', 'sig_8', 'sig_9'}';

n = length(PatientID);

% Total windows per patient
TotalWindows = [220,1280,1650,2620,2450,2050,180,1750,1620,1020,2450,1940,270,2000,990,220,2730,380,750,1900,950,1340,2190,360,1050,1250,590]';

% Artifact counts (POW, BASE, FREQ)
Art1 = [1,39,19,68,12,178,0,14,90,57,713,88,5,35,61,38,24,10,4,21,51,99,9,4,157,113,12]';
Art2 = [0,169,16,136,11,45,0,817,43,22,258,8,141,215,86,10,41,9,16,1,7,18,9,5,15,2,215]';
Art3 = [2,74,105,154,58,281,6,271,660,206,1075,167,8,223,125,73,66,6,54,27,369,152,50,18,474,180,51]';

ArtifactCounts = [Art1, Art2, Art3];

% Extract and sort numeric IDs
PatientNum = str2double(regexprep(PatientID, 'sig_', ''));
[PatientNumSorted, sortIdx] = sort(PatientNum);

% Sort all data
PatientID = PatientID(sortIdx);
PatientNum = PatientNumSorted;
TotalWindows = TotalWindows(sortIdx);
ArtifactCounts = ArtifactCounts(sortIdx, :);

% Compute per-patient artifact type weight (percentage)
weights = 100 * ArtifactCounts ./ sum(ArtifactCounts, 1);


% Plot
n = length(PatientID);
figure('Color', 'w', 'Position', [100, 100, 1300, 600]);
hb = bar(weights, 'grouped');
title('Artifact weights per patient');
xlabel('Patient ID');
ylabel('Weight (% of artifact windows per patient)');
xticks(1:n);
xticklabels(PatientNum);
xtickangle(45);
grid on;

% Bar colors
set(hb(1), 'FaceColor', [0.2 0.6 1]);   % POW
set(hb(2), 'FaceColor', [1.0 0.7 0.0]); % BASE
set(hb(3), 'FaceColor', [0.8 0.2 0.2]); % FREQ

% Add threshold line
yline(15, 'r--', '15% threshold', ...
    'Color', 'r', ...
    'LabelHorizontalAlignment', 'right', ...
    'LabelVerticalAlignment', 'bottom', ...
    'LineWidth', 1.5);

% Annotate values on bars and highlight if > threshold
threshold = 15;
for i = 1:n
    for j = 1:3
        val = weights(i, j);
        if val > 0
            x = i + (j - 2) * 0.25; % shift label within group
            y = val + 0.5;
            %y = val + 2 + 1.5*(j-2); % add variable offset for clarity
            if val > threshold
                color = 'r';
            else
                color = 'k';
            end
            text(x, y, sprintf('%.1f%% (%d)', val, ArtifactCounts(i, j)), ...
                'Color', color, ...
                'FontSize', 8, ... %'VerticalAlignment', 'top', ...
                'Rotation',90);
        end
    end
end

legend({'POW', 'BASE', 'FREQ', 'Threshold'}, 'Location', 'best');
