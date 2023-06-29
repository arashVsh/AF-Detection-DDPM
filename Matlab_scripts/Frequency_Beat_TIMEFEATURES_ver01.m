function [feature] = Frequency_Beat_TIMEFEATURES_ver01(stack, sampleRate)
%%
% INPUT:
% stack
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

ftype = 'bandpass';
n = 6;
Wn1 = [1 14]/(sampleRate/2);
[bfilter1, afilter1] = butter(n,Wn1,ftype);  

Wn2 = [8 28]/(sampleRate/2);
[bfilter2, afilter2] = butter(n,Wn2,ftype);  

Wn3 = [15 38]/(sampleRate/2);
[bfilter3, afilter3] = butter(n,Wn3,ftype);  

Wn4 = [25 68]/(sampleRate/2);
[bfilter4, afilter4] = butter(n,Wn4,ftype);  

Wn5 = [40 120]/(sampleRate/2);
[bfilter5, afilter5] = butter(n,Wn5,ftype);  
%

stack1 = filtfilt(bfilter1, afilter1, stack);
stack2 = filtfilt(bfilter2, afilter2, stack);
stack3 = filtfilt(bfilter3, afilter3, stack);
stack4 = filtfilt(bfilter4, afilter4, stack);
stack5 = filtfilt(bfilter5, afilter5, stack);
%
f1 = skewness (cumsum(stack1));
f2 = skewness (cumsum(stack2));
f3 = skewness (cumsum(stack3));
f4 = skewness (cumsum(stack4));
f5 = skewness (cumsum(stack5));
f6 = median(stack5)/median(stack1);


feature = [f1 f2 f3 f4 f5 f6];
