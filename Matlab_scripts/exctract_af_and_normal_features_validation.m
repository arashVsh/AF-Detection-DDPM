clear all
close all
labels=readtable("validation\REFERENCE-v3.csv");
labelTagTable=labels(:,2);
sampleNumber=labels(:,1);
% convert to specific type
labelCell=table2array(labelTagTable);
labelTag=cell2mat(labelCell);
sampleCell=table2array(sampleNumber);
sampleTag=cell2mat(sampleCell);
sampleString=string(sampleNumber.(1));

%% find normal and af ECG file Names
normalIndexes=[];
afIndexes=[];
for i = 1:length(labelTag)
    i
    if labelTag(i) == 'N'
        normalIndexes=[normalIndexes i];
    elseif labelTag(i) == 'A'
        afIndexes=[afIndexes i];
    end
end
%% load af data and pass it to rr-int
sampleRate=300;
AfRPeaks={};
NormalRPeaks={};
NormalRrInterval=[];
AfRrInterval=[];
NormalFeatures=[];
AfFeatures=[];
for j=1:length(afIndexes) % loop for af signals
    j
    % load data
    path=strcat("validation/",sampleString(afIndexes(j))); % concatenate string for path variable
    tmp=load(path);
    len=length(tmp.val);
    af_single_features=feature_extraction_wrapper(tmp.val);
    AfFeatures(j,:)=af_single_features;
end
%%
%% load af data and pass it to rr-int

for k=1:length(normalIndexes) % loop for af signals
    k
    % load data
    path=strcat("validation/",sampleString(normalIndexes(k))); % concatenate string for path variable
    tmp=load(path);
    len=length(tmp.val);
    normal_single_features=feature_extraction_wrapper(tmp.val);
    NormalFeatures(k,:)=normal_single_features;
end