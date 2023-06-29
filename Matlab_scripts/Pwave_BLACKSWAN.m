

function [feature_P, P] = Pwave_BLACKSWAN(xd1, R_i2, sampleRate)
%%
% INPUT:
% xd1
% R_i2
% sampleRate
%
% OUTPUT:
% feature_P
% P
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
% ============================= //////////////////////////// ============================
for i1 = 2 : length((R_i2))-1
    Endd = R_i2(i1) - ceil(0.06*sampleRate);
    Startt =  Endd - ceil(0.25*(R_i2(i1) - R_i2(i1-1))) ;
    stack = xd1(Startt : Endd);
    P{i1-1} = stack;
    stack = [];
end

for i1 = 1:length(P)
    % -----------------------------------------------------------------------------------------------------
    stack = P{i1};
    % -----------------------------------------------------------------------------------------------------
    stack1 = [];
    [M(i1), I] = max(stack);
    if I-5-1>=1 && I+5<=length(stack)
        for i = -5:5
           stack1 = [stack1, stack(I+i) - stack(I+i-1)]; 
        end
    end
    slope_P(i1) = 0.1*sum(stack1);
    stack = [];
end

% --------------------------------------------------------------------------------------------------------
f1= var(slope_P);
% --------------------------------------------------------------------------------------------------------
feature_P = [f1]; %1



