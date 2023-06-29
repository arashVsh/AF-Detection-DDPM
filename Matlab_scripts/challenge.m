function [classifyResult] = challenge(recordName)
%%
% INPUT:
% recordName      ----> input ECG signal name
%
% OUTPUT:
% classifyResult  ----> Classification result(s)
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
%% Parameters---------------------------------------------------------------------------------------------------------
sampleRate = 300; % Hz
rng(0);

load('Model.mat')

filename = sprintf('%s.mat',recordName);
load(filename)

%% Feature Extraction------------------------------------------------------------------------------------------------
% ---------------------------------------------------------------------
[feature] = feature_extraction_BLACKSWAN(val, sampleRate);
% ---------------------------------------------------------------------
indx = [1,2,3,5,7,9,11,13,15,17,21,23,25,27,28,29,30,31,32,48,52,53,54,55,56,57,58,59,61,63,65,67,69,71,73,75,78,80,81,91,97,101,104,105,108,109,110,111,112,113,114,119,120,121,122,129,139,181,182,187,188,189,190,193,194,197,198,199,200,201,202,203,204,205,207,208,209,210,211,212,213,214,215,216,218,219,220,221,222,223,232,234,242,245,246,247,250,252,253,255,257,260,261,264,265,270,271,273,275,276,280,281,282,283,284,287,290,293,294,295,296,300,305,306,325,326,327,328,329,330,331,332,333,334,341,342,343,360,361,377,399,426,444,445,446,448,454,462,463,473];
[FEATURE_validation] = [feature(:,indx)]; %#ok<NBRAK> %
%% Classification ---------------------------------------------------------
[classifyResult] = classification_BLACKSWAN(cens, NAN_MEAN_INPUT, FEATURE_validation);
[classifyResult] = convert_label_BLACKSWAN(classifyResult);
