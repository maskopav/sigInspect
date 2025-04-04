function saveResultsToExcel(excelFile, sheetName, resultsTable)
    % Check if the file already exists
    if isfile(excelFile)
        % Read existing data
        try
            existingData = readtable(excelFile, 'Sheet', sheetName);
        catch
            existingData = table(); % If sheet doesn't exist, create an empty table
        end
        % Append new results to the existing data
        updatedTable = [existingData; resultsTable]; 
    else
        % If the file doesn't exist, just use the new results
        updatedTable = resultsTable;
    end
    
    % Write the updated table to Excel
    writetable(updatedTable, excelFile, 'Sheet', sheetName);
end