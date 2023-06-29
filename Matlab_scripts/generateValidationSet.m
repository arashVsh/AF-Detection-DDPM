% This script will verify that your code is working as you intended, by
% running it on a small subset (300 records) of the training data, then
% comparing the answers.txt file that you submit with your entry with
% answers produced by your code running in our test environment using
% the same records.
%
% In order to run this script, you should have downloaded and extracted
% the validation set into the directory containing this file.
%
%
% Written by: Chengyu Liu and Qiao Li January 20 2017 
%             chengyu.liu@emory.edu  qiao.li@emory.edu
%
% Last modified by:
%%
% Modified by:  
% Contact:
% Morteza Zabihi (morteza.zabihi@gmail.com) & Ali Bahrami Rad(abahramir@gmail.com)
% Black Swan Team (April 2017)
%=======================================================================
%%
%% Feature Extraction on the Validation Set
rng(0)
total_time = 0;
addrress = cd;
[txt] = datahandling('validation', addrress); 

tic
fprintf('Classification Starts:\n')
for i = 1: length(txt{1,1})
    fprintf('Signal %d   ------------------------------------------\n', i)
    % ---------------------------------------------------------------------
    label_validation(i,:) = txt{1,1}{i,1}(end);
    % ---------------------------------------------------------------------
    recordName  = txt{1,1}{i,1}(1:end-2);
    classifyResult(i,1) = challenge(recordName);
    clear recordName
end
fprintf('\n============================================================ \n\n\n')
toc
%% Classification on the Validation Set
[Final_result] = Scoring (classifyResult, label_validation);

%% Write the classification results in a txt file
answers = dir(['answers.txt']);
if(~isempty(answers))
    display(['Found previous answer sheet file in: ' pwd])
    cont = input('Delete it (Y/N)?','s');
    if(strcmp(cont,'Y')) 
        display('Removing previous answer sheet.')
        warning('off','all'); delete answers.txt
        fid = fopen('answers.txt','wt');
        for i=1:length(classifyResult)
            fprintf(fid,'%s,%s\n',txt{1,1}{i,1}(1:end-2),classifyResult(i));
        end
        fclose(fid);
    else 
        display('New answer sheet is not made')
    end    
else
    fid = fopen('answers.txt','wt');
    for i=1:length(classifyResult)
        fprintf(fid,'%s,%s\n',txt{1,1}{i,1}(1:end-2),classifyResult(i));
    end
    fclose(fid);
end        

%% Zip
delete('entry.zip')
zip('entry.zip',{'*.m','*.mat','*.txt','*.sh'});
