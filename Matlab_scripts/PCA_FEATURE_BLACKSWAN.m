function [feature_pca]  = PCA_FEATURE_BLACKSWAN(x, R_i2, TW)
%%
% INPUT:
% x
% R_i2
% TW
% 
% OUTPUT:
% feature_pca
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
fs = 300;

 if length(R_i2)<=3
        X_FeaturesSpace1 = NaN*ones(1,32);
        X_FeaturesSpace2 = NaN*ones(1,32);
 else
         f = 1;
         [b,a] = butter(3, 2*f/fs, 'high'); 
         x = filtfilt(b,a,x); 
 
        for i=2:length(R_i2)-1
            remin1 = [];
            remin2 = [];
            lim1 = floor(R_i2(i) - TW);
            if lim1<1
                remin1 = zeros(1,abs(lim1)+1);
                lim1 = 1;
            end
            lim2 = floor(R_i2(i) + TW);
            if lim2 >length(x)
                remin2 = zeros(1,abs(lim2-length(x)));
                lim2 = length(x);
            end
            data1(i-1,:) = [remin1, x(lim1:lim2), remin2];    

        end
        MEANdata1 = mean(data1,1);
        STDdata1 = std(data1,1);
        X_FeaturesSpace1 = waveletECG1_BLACKSWAN(MEANdata1);  
        X_FeaturesSpace2 = waveletECG1_BLACKSWAN(STDdata1);
 end
 %%
 if length(R_i2)<=10
    feature_eig = NaN*ones(1,8);
 else
    [PC_dir,PC_val,eigen_val] = pca(data1); 
    eigen_val1 = eigen_val/eigen_val(1);
    eigen_val1(1) = eigen_val(1);  
    feature_eig = eigen_val1(1:8)'; 
 end
feature_pca =  [feature_eig, X_FeaturesSpace1,  X_FeaturesSpace2];
