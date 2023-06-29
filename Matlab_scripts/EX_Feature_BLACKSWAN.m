
function [FEATURE] = EX_Feature_BLACKSWAN(x, R_i2, sampleRate)
%%
% INPUT:
% X
% R_i2
% sampleRate
%
% OUTPUT:
% feature
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
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if length(R_i2)>4
    RR = diff(R_i2')/sampleRate;
    RR1 = RR(1:end-1);
    RR2 = RR(2:end);

    factor = (1/(length(RR)-2));
    nom = factor*sum (  (diff(RR1(end:-1:1)).^2) + (diff(RR2(end:-1:1)).^2) );
    dnom = mean(RR);
    stepping = nom/dnom; %***
else
    stepping = NaN; %1
end
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
w2 = NaN;
if length(R_i2)>3
    RR = diff(R_i2')/sampleRate;
    RRR = buffer(RR, 3, 0);
    RRR(:,[1, size(RRR,2)]) = [];

    RRR1 = sum(RR)./60;
    indx = find(RRR1 < 46);
    w2 = RRR1(indx);     %#ok<FNDSB>
end
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
f1 = NaN; f2 = NaN; f3 = NaN; f4 = NaN; f5 = NaN; f6 = NaN; f7 = NaN;
if length(R_i2)>6
    beat = beat_beat_BLACKSWAN(x, R_i2);
    data  = Equalization_BLACKSWAN (beat, 150);
    
    f = data(:,30:40);                     f1 = sum((std(f)).^2);        clear f
    f = data(:,40:50);                     f2 = sum((std(f)).^2);        clear f
    f = data(:,50:60);                     f3 = sum((mean(f)).^2);     clear f
    f = data(:,60:70);                     f4 = sum((std(f)).^2);        clear f
    f = data(:,80:90);                     f5 = sum((std(f)).^2);        clear f
    f = data(:,90:100);                   f6 = sum((mean(f)).^2);     clear f
    f = data(:,100:110);                 f7 = sum((std(f)).^2);        clear f
      
end
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
beat = buffer(x, 4*sampleRate, 2*sampleRate);
 for jj =1: size(beat,2)
        a = beat(:,jj);
        CH(jj) = CheckQuality_BLACKSWAN(a, sampleRate);
 end
 
s1 = length(find(CH ==1)) / length(CH);
s2 = length(find(CH ==2)) / length(CH);
%%
FEATURE = [stepping, mean(w2), f1 f2 f3 f4 f5 f6 f7 s1 s2]; 

