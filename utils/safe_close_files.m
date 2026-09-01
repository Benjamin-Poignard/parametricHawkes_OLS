function safe_close_files(fid)

try
    if fid > 0
        fclose(fid);
    end
catch
end
