function [labels_1] = convert_label_BLACKSWAN(label)
%%
% INPUT:
% label      ---->
%
% OUTPUT:
% labels_1  ----> 
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
if ischar( label(1,:)) == 1

    for i = 1 : length(label)
        if label(i) ==  'N'
            labels_1(i,1) =1;
        elseif label(i) ==  'A'
            labels_1(i,1) =2;
        elseif label(i,1) ==  'O'
            labels_1(i,1) =3;
        elseif label(i) ==  '~'
            labels_1(i,1) =4;
        end
    end

% --------------------------------------------------------------------------------------------------------   
else
% --------------------------------------------------------------------------------------------------------

    for i = 1 : length(label)
        if label(i) == 1
            labels_1(i,1) = 'N';
        elseif label(i) == 2
            labels_1(i,1) = 'A';
        elseif label(i) == 3
            labels_1(i,1) = 'O';
        elseif label(i) == 4
            labels_1(i,1) = '~';
        end
    end

end
    