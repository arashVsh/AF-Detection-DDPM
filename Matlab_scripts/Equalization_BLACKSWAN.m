

function [data]  = Equalization_BLACKSWAN(beat, sn)
%%
% INPUT:
% beat 
%
% OUTPUT:
% data
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
for jj =1: size(beat,2)
    a = beat{1,jj};
    xj = 1 : numel(a);
    xp = linspace(xj(1), xj(end), sn);
    y = interp1(xj, a, xp);
    data(jj,:) = y;
    clear y a cj xp
end