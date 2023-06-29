function [PWAVE] = CheckPWAVE_BLACKSWAN(AAA, sampleRate)
%%
% INPUT:
% AAA          ----> 
% sampleRate ----> 
%
% OUTPUT:
% PWAVE     ---->
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
[QRS, ~, ~] = qrs_detect21(AAA,0.25,0.6,sampleRate); 
R_i2 = QRS;
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if length(R_i2) > 2
    k = 1;
    for i1 = 2 : length((R_i2))-1
        Endd = R_i2(i1) - ceil(0.06*sampleRate);
        Startt =  Endd - floor(  (R_i2(i1) - R_i2(i1-1))/2) - ceil(0.06*sampleRate);
        
        Pwave = AAA(Startt : Endd);
        xj = 1 : numel(Pwave); xp = linspace(xj(1), xj(end), 128);
        PWAVE(k,:) = interp1(xj, Pwave, xp);
        k = k + 1;
        clear Endd Startt Pwave xj xp
    end
else
    PWAVE = [];
end
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

