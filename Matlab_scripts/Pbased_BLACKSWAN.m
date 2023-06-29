
function [Pbased_FEATURE] = Pbased_BLACKSWAN(x, sampleRate)
%%
% INPUT:
% x
% sampleRate
%
% OUTPUT:
% DFA_FEATURE
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
cutOffFreq1 = 0.5; filterOrder1 = 2;
[b1, a1] = butter(filterOrder1, cutOffFreq1/(sampleRate/2),'high');
% x = filtfilt(b1, a1, x); % Apply filter to data using zero-phase filtering / Highpass
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
nn = 5;
beat = buffer(x, nn*sampleRate, 4*sampleRate);
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
PWAVE = [];
for jj = nn: size(beat,2)
    a = []; a = beat(:,jj);
    AAA = smooth(filtfilt(b1, a1, a)); 
    PWAVE = [PWAVE; CheckPWAVE_BLACKSWAN(AAA, sampleRate)];
end
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if ~isempty(PWAVE)
    W1 = PWAVE;
    [NN, MM] =size(W1);
    
    xbins1 = 0:64;
    [counts, centers] = hist(W1(:), xbins1);
    
    
    thr = mean((diff(centers))/2);
    centers(2,:) = centers(1,:) + thr;
    centers(3,:) = centers(1,:) - thr;
    centers(3,1) = 0;
    
    P = zeros(NN, MM);
    for i = 1:64
        ind  = (W1>= centers(3,i)  &W1< centers(2,i));
        P(ind)  = counts(i)/sum(counts);
    end
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*    
    for i=1:NN
        for j= 1:MM
            P1(i,j) = P(i,j)/1+abs(i-j);
        end
    end
    
    F1 = sum(P1(:))/(NN*MM);
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    P(P == 0) = 0.00000000000000001;
    [~, ~, Renyi] = entropy_BLACKSWAN(P(:), 2);
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    F22 = mean(std(PWAVE));
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    U = corrcoef(PWAVE); U = triu(U); U1= sum(U(:)) - size(U, 2); F33 = U1/size(PWAVE, 1);
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    U = U(:); U(U == 0) = []; F44 = std(U); 
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    F55 = max(mean(PWAVE));
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

else
    F1 = NaN;
    Renyi = NaN;
    F22 = NaN;
    F33 = NaN;
    F44 = NaN;
    F55 = NaN;
end

% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
Pbased_FEATURE = [F22 F33 F44 F55 F1 Renyi];




