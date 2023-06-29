function [feature_T, T] = Twave__BLACKSWAN(xd1, R_i2, sampleRate)

%%
% INPUT:
% xd1
% R_i2
% sampleRate
%
%
% OUTPUT:
% feature_T  
% T
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
% ============================= /////////////////////////////// ===========================
for i1 = 2 : length((R_i2))-1
    Startt = R_i2(i1) + floor(0.06*sampleRate); 
    Endd =  R_i2(i1) + floor(0.5*(R_i2(i1+1) - R_i2(i1))) ; % 
    stack = xd1(Startt : Endd);
    if isempty(stack)
        stack = 0;
    end
    T{i1-1} = stack;
    % --------------------------------------------------------------------------------------------------------
    % --------------------------------------------------------------------------------------------------------
    kurtosis_T(i1) = kurtosis(stack);
    % --------------------------------------------------------------------------------------------------------
    % --------------------------------------------------------------------------------------------------------
    stack = [];
end


% --------------------------------------------------------------------------------------------------------
f1 = mean(kurtosis_T);

feature_T  = [f1];