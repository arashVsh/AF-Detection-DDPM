
function [feature_2]  = faeture_extraction_ver1_ind1_BLACKSWAN(beat)
%%
% INPUT:
% beat 
%
% OUTPUT:
% feature_2
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
level = 5;
waven = 'sym4';
q = 2;

    
for i1 = 1:length(beat)
    clear x
    x = beat{i1};
    % -----------------------------------------------------------------------------------------------------
    clear c l a5 det5 det4 det3
    [c,l] = wavedec(x, level, waven);
    
    a5 = appcoef(c,l,waven);
    det5 = detcoef(c,l,level);
    det4 = detcoef(c,l,level-1);
    % -----------------------------------------------------------------------------------------------------
    [p] = probability_BLACKSWAN(x);
    [Shannon(i1) , Tsallis(i1) , Renyi(i1) ] = entropy_BLACKSWAN(p, q);
    % -----------------------------------------------------------------------------------------------------
    clear p
    [p] = probability_BLACKSWAN(a5);
    [Shannon_a5(i1) , Tsallis_a5(i1) , Renyi_a5(i1) ] = entropy_BLACKSWAN(p, q);
    % -----------------------------------------------------------------------------------------------------
    clear p
    [p] = probability_BLACKSWAN(det5);
    [Shannon_d5(i1) , Tsallis_d5(i1) , Renyi_d5(i1) ] = entropy_BLACKSWAN(p, q);
    % -----------------------------------------------------------------------------------------------------
    clear p
    [p] = probability_BLACKSWAN(det4);
    [Shannon_d4(i1) , Tsallis_d4(i1) , Renyi_d4(i1) ] = entropy_BLACKSWAN(p, q);
    % -----------------------------------------------------------------------------------------------------
    
end
   
%%
Shannon_1 = mean(Shannon);
Shannon_2 = var(Shannon);

Tsallis_1 = mean(Tsallis);
Tsallis_2 = var(Tsallis);   

Renyi_1 = mean(Renyi);
Renyi_2 = var(Renyi);
% =====================================================================
Shannon_a5_1 = mean(Shannon_a5);
Shannon_a5_2 = var(Shannon_a5);
    
Tsallis_a5_1 = mean(Tsallis_a5);
Tsallis_a5_2 = var(Tsallis_a5);

Renyi_a5_1 = mean(Renyi_a5);
Renyi_a5_2 = var(Renyi_a5);
% =====================================================================
Shannon_d5_1 = mean(Shannon_d5);
Shannon_d5_2 = var(Shannon_d5);

Tsallis_d5_1 = mean(Tsallis_d5);
Tsallis_d5_2 = var(Tsallis_d5);

Renyi_d5_1 = mean(Renyi_d5);
Renyi_d5_2 = var(Renyi_d5); 
% =====================================================================
Shannon_d4_1 = mean(Shannon_d4);
Shannon_d4_2 = var(Shannon_d4);

Tsallis_d4_1 = mean(Tsallis_d4);
Tsallis_d4_2 = var(Tsallis_d4);

Renyi_d4_1 = mean(Renyi_d4);
Renyi_d4_2 = var(Renyi_d4);



feature_2 = [Shannon_1 Shannon_2 Tsallis_1 Tsallis_2 Renyi_1 Renyi_2,...
                   Shannon_a5_1 Shannon_a5_2 Tsallis_a5_1  Tsallis_a5_2 Renyi_a5_1 Renyi_a5_2,...
                   Shannon_d5_1 Shannon_d5_2 Tsallis_d5_1 Tsallis_d5_2 Renyi_d5_1 Renyi_d5_2,...
                   Shannon_d4_1 Shannon_d4_2 Tsallis_d4_1 Tsallis_d4_2 Renyi_d4_1 Renyi_d4_2]; 