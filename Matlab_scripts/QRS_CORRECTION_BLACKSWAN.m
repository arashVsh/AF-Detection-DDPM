
function [R_i1] = QRS_CORRECTION_BLACKSWAN(x, QRS)
%%
% INPUT:
% x
% QRS
%
% OUTPUT:
% R_i1
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


R_i2 = []; R_i2 = QRS;  R_amp2 = x(QRS);
kw = 5;
for i1 = 1:length(R_i2)
    
    if R_i2(i1)-kw>=1 && R_i2(i1)+kw<= length(x)
        
        stack = []; stack = x(R_i2(i1)-kw : R_i2(i1)+kw);
        M  = []; I  = []; [M,I] = max(abs(stack));
        
        R_amp1(i1) = M;
        R_i1(i1) = I + (R_i2(i1)-kw)-1;
    else
        R_amp1(i1) = R_amp2(i1);
        R_i1(i1) = R_i2(i1);
    end
    
end


h = mode(sign(x(QRS)));
indx = sign(x(R_i1));
indx2 = find(indx ~=h);

R_i1(indx2) = QRS(indx2);







