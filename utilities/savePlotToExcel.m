function savePlotToExcel(figHandle, excelFile, sheetName, imageFile)
    % Save the figure as a temporary image
    exportgraphics(figHandle, imageFile, 'Resolution', 300);

    % Open Excel via COM
    excel = actxserver('Excel.Application');
    excel.Visible = false;  % Set to true to see Excel opening

    % Open or create the workbook
    if isfile(excelFile)
        workbook = excel.Workbooks.Open(fullfile(pwd, excelFile));
    else
        workbook = excel.Workbooks.Add;
        workbook.SaveAs(fullfile(pwd, excelFile));
    end

    % Delete the sheet if it already exists
    try
        sheet = excel.Worksheets.Item(sheetName);
        sheet.Delete;
    catch
        % Sheet doesn't exist — do nothing
    end

    % Add new sheet and rename
    sheet = workbook.Sheets.Add([], workbook.Sheets.Item(workbook.Sheets.Count));
    sheet.Name = sheetName;

    % Insert image into the sheet
    shapes = sheet.Shapes;
    shapes.AddPicture(fullfile(pwd, imageFile), 0, 1, 10, 10, 600, 400);

    % Save and close
    workbook.Save();
    workbook.Close();
    excel.Quit();
    delete(excel);
end
