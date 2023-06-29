
function [X2] = baselinewanderremoval_BLACKSWAN(x)
%%
% INPUT:
% x 
%
% OUTPUT:
% X2
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
fc = 1/300; 
d = 1;        
r = 6; 
amp = 0.8;      
lam0 = 0.5*amp;
lam1 = 5*amp;
lam2 = 4*amp;

XQ = buffer(x, 300, 0);


for i = 1:size(XQ,2)
    [x1(:,i), ~, ~] = beads(XQ(:,i), d, fc, r, lam0, lam1, lam2);
end

X2  =  reshape(x1, size(x1,1)*size(x1,2), 1);
