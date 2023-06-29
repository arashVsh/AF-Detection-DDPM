function [CH] = CheckQuality_BLACKSWAN(AAA, sampleRate)
%%
% INPUT:
% AAA          ----> Raw data
% sampleRate ----> 
%
% OUTPUT:
% CH              ---->
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
[QRS, ~, ~] = qrs_detect2(AAA,0.25,0.6,sampleRate); % Detect the QRS picks
R_i2 = QRS;
% *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
if length(R_i2) > 3
    k = 1;
    for i1 = 2 : length((R_i2))-1
        Endd = R_i2(i1) - ceil(0.06*sampleRate);
        Startt =  Endd - floor(  (R_i2(i1) - R_i2(i1-1))/2);

        Endd1 = R_i2(i1) + ceil(0.06*sampleRate);

        VAR_QRS(k) = var(AAA(Endd : Endd1));
        VAR_BQRS(k) = var(AAA(Startt : Endd));
         k = k + 1;
    end
    % *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    indx  = VAR_QRS - VAR_BQRS;
    if ~isempty(find(indx<=0)) %#ok<EFIND>
        CH = 2; % noisy
    else
        CH = 1; % not noisy
    end
else
    CH = 2;
end

