function aTag = alpha_tag(a)

% alpha-level tag for field names

aTag = sprintf('alpha%g',a);
aTag = strrep(aTag,'.','p');
aTag = strrep(aTag,'-','m');