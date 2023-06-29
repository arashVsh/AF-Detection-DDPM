function X_FeaturesSpace = morphoECG1_BLACKSWAN(X)
%%
% INPUT:
% x         
% 
% OUTPUT:
% X_FeaturesSpace
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

features = [];
for i = 1:size(X,1)
    
 x = X(i,:);
 
 
 PAR = sum(0.5 * (x + abs(x)));
 
 NAR = sum(0.5 * (x - abs(x)));
 
 TAR = PAR + NAR;
 
 ATAR = abs(TAR);
 
 TAAR = PAR + abs(NAR);
 
 d_x = diff(x);
 
 AASS = mean(abs(d_x));
 
 PP = max(x) - min(x);
 
 d_flip_x = diff(fliplr(x));
 
 SSA = sum(0.5 * abs( d_flip_x/abs(d_flip_x) + d_x / abs(d_x)));
 
 
 
 features1 = [PAR, NAR, PP];


features = [features; features1];




end

X_FeaturesSpace = features;


 