function [p] = probability_BLACKSWAN(x, nbins)
%%
% INPUT:
% x
% nbins
%
%
% OUTPUT:
% p  ----> probability
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
switch nargin
    case 1
     nbins = 30;
end    
    
% --------------------------------------------------------------------------------------------------------
a = min(x);
if a<0
    x = x + abs(a);
end
% --------------------------------------------------------------------------------------------------------
[counts,centers] = hist(x,nbins);
thr = mean((diff(centers))/2);
centers(2,:) = centers(1,:) + thr;
centers(3,:) = centers(1,:) - thr;
centers(3,1) = 0;

for i = 1 : length(x)
    p1 = [];
    p1 = find(x(i) <= centers(2,:) & x(i) >= centers(3,:) );
    if isempty (p1)
            p(i) = counts(end);
    else
            p(i) = counts(p1(1));
    end
    clear p1

    if p(i)  ==0
            p(i) = 0.00001;
    end
end
p = p / sum(p);
% --------------------------------------------------------------------------------------------------------
