
function [feature_freq] = frequency_feature_BLACKSWAN(beat, sampleRate)
%%
% INPUT:
% beat
% sampleRate
%
% OUTPUT:
% feature_freq
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

for i=1:length(beat)
    xd2 = beat{i};
    [pxx, f] = pburg(xd2,11,[], sampleRate); 

    ind1 = find(5<=f & f<=15);
    ind2 = find(5<=f & f<=40);
    ind3 = find(1<=f & f<=40);
    ind4 = find(0<=f & f<=40);

    f3 = sum(pxx(ind1))/sum(pxx(ind2)); %#ok<FNDSB>
    f4 = sum(pxx(ind3))/sum(pxx(ind4)); %#ok<FNDSB>

    feature1(i,:) = [f3 f4]; %#ok<AGROW>
    clear ind1 ind2 ind3 ind4 f3 f4
end


feature_freq = mean(feature1);
