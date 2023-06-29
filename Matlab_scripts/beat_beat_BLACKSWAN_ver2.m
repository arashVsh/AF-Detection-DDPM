function [beat] = beat_beat_BLACKSWAN_ver2(xd1, R_i2)
%%
% INPUT:
% xd1 
% R_i2 
%
% OUTPUT:
% beat  ---->  Cell
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
for i1 = 2 : length((R_i2))-1
    Endd = floor( (R_i2(i1+1) - R_i2(i1)) /2);
    Startt =  floor((R_i2(i1) - R_i2(i1-1))/2) ;
    stack = xd1(R_i2(i1)-Startt : R_i2(i1)+Endd);
    beat{1,i1-1} = stack;
    beat{2, i1-1} = R_i2(i1) - (R_i2(i1) - Startt) +1; %#ok<AGROW>
    
    
end

