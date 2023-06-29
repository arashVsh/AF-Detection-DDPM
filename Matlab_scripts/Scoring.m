function [Score] = Scoring (classifyResult, label_validation)
%%
% INPUT:
% classifyResult
% label_validation
%
% 
% OUTPUT:
% Score
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
% true_label_val = label_validation;
[predicted_label_val] = convert_label_BLACKSWAN(classifyResult);
[true_label_val] = convert_label_BLACKSWAN(label_validation);

[ldaResubCM, ~] = confusionmat(true_label_val,  predicted_label_val);

%% Scoring
Nn = ldaResubCM(1,1);   Na = ldaResubCM(1,2);  No = ldaResubCM(1,3);   Np = ldaResubCM(1,4);
An = ldaResubCM(2,1);   Aa = ldaResubCM(2,2);   Ao = ldaResubCM(2,3);   Ap = ldaResubCM(2,4);
On = ldaResubCM(3,1);   Oa = ldaResubCM(3,2);  Oo = ldaResubCM(3,3);   Op = ldaResubCM(3,4);
Pn = ldaResubCM(4,1);    Pa = ldaResubCM(4,2);   Po = ldaResubCM(4,3);   Pp = ldaResubCM(4,4);

sigma_n = sum(ldaResubCM(:,1));     sigma_N = sum(ldaResubCM(1,:));
sigma_a = sum(ldaResubCM(:,2));     sigma_A = sum(ldaResubCM(2,:));
sigma_o = sum(ldaResubCM(:,3));     sigma_O = sum(ldaResubCM(3,:));
sigma_p = sum(ldaResubCM(:,4));     sigma_P = sum(ldaResubCM(4,:));

F1n = (2*Nn)/(sigma_N + sigma_n);
F1a = (2*Aa)/(sigma_A + sigma_a);
F1o = (2*Oo)/(sigma_O + sigma_o);
F1p = (2*Pp)/(sigma_P + sigma_p);

F1 = (F1n + F1a + F1o ) / 3;



Score = [F1];
%% Print ------------------------------------------------------------------
fprintf('Classification Results: \n');
fprintf('======================================================================= \n');
fprintf('F1n (Normal): -----------------------  %1.4f\n',  (F1n));
fprintf('F1a (AF): ---------------------------- %1.4f\n',  (F1a));
fprintf('F1o (Other): -------------------------  %1.4f\n',  (F1o));
fprintf('F1p (Noisy): -------------------------  %1.4f\n',  (F1p));
fprintf('F1: ----------------------------------- %1.4f\n',  (F1));
fprintf('======================================================================= \n');
