function [Homogeneity] = Homogeneity_ver1_BLACKSWAN(x, sampleRate)
%%
% INPUT:
% x 
% sampleRate
%
% OUTPUT:
% Homogeneity
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
W1= cwt(x,sampleRate);
%%
[NN, MM] =size(W1);

xbins1 = 0:255;
[counts, centers] = hist(W1(:), xbins1);


thr = mean((diff(centers))/2);
centers(2,:) = centers(1,:) + thr;
centers(3,:) = centers(1,:) - thr;
centers(3,1) = 0;

P = zeros(NN, MM);
for i = 1:256
    ind  = (W1>= centers(3,i)  &W1< centers(2,i));
    P(ind)  = counts(i)/sum(counts);
end
    
    
for i=1:NN
    for j= 1:MM
        P1(i,j) = P(i,j)/1+abs(i-j);
    end
end

Homogeneity1 = sum(P1(:))/(NN*MM); 
%% 
Homogeneity =  [Homogeneity1];


