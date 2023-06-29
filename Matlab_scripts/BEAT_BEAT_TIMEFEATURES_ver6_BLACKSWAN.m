function [temp_feature2] = BEAT_BEAT_TIMEFEATURES_ver6_BLACKSWAN(x, sampleRate)
%%
% INPUT:
% x
% sampleRate
%
% OUTPUT:
% temp_feature2
%=======================================================================
% Authors' contact:
% Morteza Zabihi (morteza.zabihi@gmail.com) & Ali Bahrami Rad(abahramir@gmail.com)
% Black Swan Team (April 2017)
%=======================================================================
% This code is released under the GNU General Public License.
% You should have received a copy of the GNU General Public License along with this program.
% If not, see <http://www.gnu.org/licenses/>.

% By accessing the code through Physionet webpage and/or by installing, copying, or otherwise
% using this software, you agree to be bounded by the terms of GNU General Public License.
% If you do not agree, do not install copy or use the software.

% We make no representation about the suitability of this licensed deliverable for any purppose.
% It is provided "as is" without express or implied warranty of any kind.
% Any use of the licensed deliverables must include the above disclaimer.
% For more detailed information please refer to "readme" file.
%%
rng(0)
cutOffFreq1 = 1.1; filterOrder1 = 2;
[bbfilter, aafilter] = butter(filterOrder1, cutOffFreq1/(sampleRate/2),'high');
%% ------------------------------------------------------------------------------------------------------------------------------
x = smooth(filtfilt(bbfilter, aafilter, x));
x = detrend(x);
x = x - mean(x);
%% ------------------------------------------------------------------------------------------------------------------------------
[QRS, ~, ~] = qrs_detect21(x,0.25,0.6,sampleRate);
R_i2 = QRS;
%% ------------------------------------------------------------------------------------------------------------------------------
stack = x;
step1 = 6;
if  length(R_i2)>5
    
    for i  = 2 : length((R_i2))-2

        Rloc = R_i2(i);
        limit1 = Rloc - 75;
        limit2 = Rloc + 135;

        tbefore =Rloc:-step1:limit1; tbefore = tbefore(end:-1:1);
        tafter = Rloc+step1:step1:limit2;
        
        Before = stack(Rloc:-step1:limit1); Before = Before(end:-1:1);
        After = stack(Rloc+step1:step1:limit2);
        
        
        new_sig(i-1,:) = [Before', After'];

        clear Before After Rloc limit1 limit2 tbefore tafter
    end
    
else
        new_sig = NaN*ones(1, 35);
end
    
clear stack

%% ------------------------------------------------------------------------------------------------------------------------------
stack = [0; diff(x)];

if  length(R_i2)>5
    
    for i  = 2 : length((R_i2))-2

        Rloc = R_i2(i);
        limit1 = Rloc - 75;
        limit2 = Rloc + 135;

        tbefore =Rloc:-step1:limit1; tbefore = tbefore(end:-1:1);
        tafter = Rloc+step1:step1:limit2;

        Before = stack(Rloc:-step1:limit1); Before = Before(end:-1:1);
        After = stack(Rloc+step1:step1:limit2);
        new_sig1(i-1,:) = [Before', After'];

        clear Before After Rloc limit1 limit2 tbefore tafter
    end
else
    new_sig1 = NaN*ones(1, 35);
end

clear stack step1
%% ------------------------------------------------------------------------------------------------------------------------------
stack = [0; 0; diff(diff((x)))];
step1 = 1;
if length(R_i2)>5
    
    for i  = 2 : length((R_i2))-2

        Rloc = R_i2(i);
        limit1 = Rloc - 25;
        limit2 = Rloc + 15;

        tbefore =Rloc:-step1:limit1; tbefore = tbefore(end:-1:1); tbefore1 = tbefore(1:2:end-10); tbefore2 = [tbefore1, tbefore(end-9:end)]; 
        tafter = Rloc+step1:step1:limit2;                                    tafter1 = tafter(1:2:end-10); tafter2 = [tafter1, tafter(end-9:end)]; 

        Before = stack(Rloc:-step1:limit1); Before = Before(end:-1:1); Before1 = Before(1:2:end-10); Before2 = [Before1; Before(end-9:end)]; 
        After = stack(Rloc+step1:step1:limit2);                                   After1 = After(1:2:end-10);     After2 = [After1; After(end-9:end)]; 
        new_sig2(i-1,:) = [Before2', After2'];

        clear Before After Rloc limit1 limit2 tbefore tafter Before2 After2 tafter2 tafter1 tbefore1 tbefore2
    end
    
else
        new_sig2 = NaN*ones(1, 31);
end

clear stack
%%--------------------------------------------------------------------------------------------------------------------------------
temp_feature = [new_sig,  new_sig1, new_sig2];
%%--------------------------------------------------------------------------------------------------------------------------------
temp_feature2 = [trimmean(temp_feature,10,'round',1), nanstd(temp_feature,[],1)];
