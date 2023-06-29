function [classifyResult] = classification_BLACKSWAN(cens, NAN_MEAN_INPUT, FEATURE_validation)
%%
% INPUT:
% MODEL
% FEATURE_validation    ----> Extracted Features
%
% OUTPUT:
% classifyResult                 ----> Classification result(s)
%
%
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
%% Classification ---------------------------------------------------------
rng(0);
FEATURE = FEATURE_validation;

nanInd = isnan(FEATURE);
for ii = 1:size(FEATURE,2)
    FEATURE(nanInd(:,ii),ii) = NAN_MEAN_INPUT(ii);   
end

Xtest = FEATURE;

predicted_label = predict(cens, Xtest);


predicted_label = cell2mat(predicted_label);

if predicted_label == 'NO'
    classifyResult = 1;
elseif predicted_label == 'AF'
    classifyResult = 2;
elseif predicted_label == 'OT'
    classifyResult = 3;
elseif predicted_label == 'CR'
        classifyResult = 4;
end




