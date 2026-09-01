function col = color_for_n(n)

switch n
    case 1
        col = 'r';
    case 2
        col = 'b';
    case 3
        col = 'c';
    case 4
        col = 'k';
    otherwise
        col = 'm';
end