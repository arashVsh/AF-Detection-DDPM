
function [FEATURE] = feature_extraction_BLACKSWAN(x, sampleRate)
%%
% INPUT:
% x
% sampleRate
%
% OUTPUT:
% FEATURE
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
rng(0);
warning('off','all');
load('Model_Feature1.mat')
load('Model_Feature2.mat')
load('Model_Feature3.mat')
%%
x_orig = x(:)'; x_orig(1:2*sampleRate) = [];
[QRS_orig, ~, ~] = qrs_detect21(x_orig,0.25,0.6,sampleRate); 
%-----------------------------------------------------------------------------------------------------------------------------------
[X2] = baselinewanderremoval_BLACKSWAN(x); x = []; x = X2; X2 = [];
x  = x(:)'; x(1:2*sampleRate) = [];
[QRS, ~, ~] = qrs_detect21(x,0.25,0.6,sampleRate); 
%-----------------------------------------------------------------------------------------------------------------------------------
TW  = 0.5*sampleRate;
%%
if length(QRS)<=3
    
    [faeture_HRV] = HRV_BLACKSWAN(QRS);
    feature_2 = NaN*ones(1,24);
    feature_P = NaN*ones(1,1);
    feature_3 = NaN*ones(1,24);
    feature_T = NaN*ones(1,1);
    feature_4 = NaN*ones(1,24);
    
    X_FeaturesSpace = waveletECG1_BLACKSWAN(x);

    AFEv = NaN; ATEvidence = NaN;  OrgIndex = NaN; 
    feature_freq = NaN*ones(1,2);
    
    [feature_HRV_PS]  = HRV_PS_BLACKSWAN (QRS_orig, sampleRate);
    [feature_pca]  = PCA_FEATURE_BLACKSWAN(x, QRS, TW);
    [Homogeneity] = Homogeneity_ver1_BLACKSWAN(x_orig, sampleRate);
    [FEATURE_EX] = EX_Feature_BLACKSWAN(x_orig, QRS_orig, sampleRate);
    [Feature_States, featuresoriginstates] = SegmentBased_Detection_BLACKSWAN_verPost(x, sampleRate, Model_Feature1, Model_Feature2, Model_Feature3);
    [Pbased_FEATURE] = Pbased_BLACKSWAN(x, sampleRate);
    [featuresBeat1] = BEAT_BEAT_TIMEFEATURES_ver03(x,sampleRate,QRS);
    [feature_NEWPS] = NEWPS_BLACKSWAN(x_orig, sampleRate, QRS_orig);
    [temp_feature2] = BEAT_BEAT_TIMEFEATURES_ver6_BLACKSWAN(x, sampleRate);
    
else
       
    % =====================================================================
    [beat] = beat_beat_BLACKSWAN(x, QRS);
    % =====================================================================
    [faeture_HRV] = HRV_BLACKSWAN(QRS);
    % =====================================================================
    [feature_2]  = faeture_extraction_ver1_ind1_BLACKSWAN(beat);
    % =====================================================================
    [feature_P, P] = Pwave_BLACKSWAN(x, QRS, sampleRate);
    [feature_3]  = faeture_extraction_ver1_ind1_BLACKSWAN(P);
    % =====================================================================
    [feature_T, T] = Twave__BLACKSWAN(x_orig, QRS_orig, sampleRate);
    [feature_4]  = faeture_extraction_ver1_ind1_BLACKSWAN(T); 
    % =====================================================================
    X_FeaturesSpace = waveletECG1_BLACKSWAN(x); 
    % =====================================================================
    RR = diff(QRS')/sampleRate;  
    [AFEv, ATEvidence, OrgIndex] = comput_AFEv(RR);
    % =====================================================================
    [feature_freq] = frequency_feature_BLACKSWAN(beat, sampleRate); 
    % =====================================================================
    [feature_HRV_PS]  = HRV_PS_BLACKSWAN (QRS_orig, sampleRate);
    % =====================================================================
    [feature_pca]  = PCA_FEATURE_BLACKSWAN(x, QRS, TW);  
    % =====================================================================
    [Homogeneity] = Homogeneity_ver1_BLACKSWAN(x_orig, sampleRate);
    % =====================================================================
    [FEATURE_EX] = EX_Feature_BLACKSWAN(x_orig, QRS_orig, sampleRate);
    % =====================================================================
    [Feature_States, featuresoriginstates] = SegmentBased_Detection_BLACKSWAN_verPost(x, sampleRate, Model_Feature1, Model_Feature2, Model_Feature3);
    % =====================================================================
    [Pbased_FEATURE] = Pbased_BLACKSWAN(x, sampleRate);
    % =====================================================================
    [featuresBeat1] = BEAT_BEAT_TIMEFEATURES_ver03(x, sampleRate, QRS);
    % =====================================================================
    [feature_NEWPS] = NEWPS_BLACKSWAN(x_orig, sampleRate, QRS_orig); 
    % =====================================================================
    [temp_feature2] = BEAT_BEAT_TIMEFEATURES_ver6_BLACKSWAN(x, sampleRate); 
end
%%
FEATURE = [faeture_HRV, ...
                      feature_2,...
                      feature_P,...
                      feature_3,...
                      feature_T,...
                      feature_4,...
                      X_FeaturesSpace,...
                      AFEv, ATEvidence, OrgIndex,...
                      feature_freq,...
                      feature_HRV_PS,...
                      feature_pca,...
                      Homogeneity,...
                      FEATURE_EX,...
                      Feature_States,...
                      featuresoriginstates,...
                      Pbased_FEATURE,...
                      featuresBeat1,...
                      feature_NEWPS,...
                      temp_feature2];
                  
