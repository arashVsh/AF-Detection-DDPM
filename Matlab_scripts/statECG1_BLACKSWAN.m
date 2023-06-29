function  X_FeatureSpace = statECG1_BLACKSWAN(d)
%%
% INPUT:
% d
%
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

d_diff = diff(d,1,2);
dd_diff = diff(d_diff,1,2);

MED = median(d,2);
S = std(d')';
V = var(d')';
Med = median(d,2);
MAD = mad(d')';

Kur = kurtosis(d,0,2);
Skw = skewness(d,0,2);

Kur_d = kurtosis(d_diff,0,2);
Skw_d = skewness(d_diff,0,2);

m2 = moment(d,2,2);
m3 = moment(d,3,2);
m4 = moment(d,4,2);

M2 = m2.^(1/2);
M3 = abs(m3).^(1/3);
M4 = m4.^(1/4);

m2d_diff = moment(d_diff,2,2);
m3d_diff = moment(d_diff,3,2);
m4d_diff = moment(d_diff,4,2);

M2d_diff = m2d_diff.^(1/2);
M3d_diff = abs(m3d_diff).^(1/3);
M4d_diff = m4d_diff.^(1/4);

m2dd_diff = moment(dd_diff,2,2);
m3dd_diff = moment(dd_diff,3,2);
m4dd_diff = moment(dd_diff,4,2);

M2dd_diff = m2dd_diff.^(1/2);
M3dd_diff = abs(m3dd_diff).^(1/3);
M4dd_diff = m4dd_diff.^(1/4);

IP1 = prctile(d,60,2) - prctile(d,10,2);
IP2 = prctile(d,90,2) - prctile(d,80,2);
IP3 = prctile(d,80,2) - prctile(d,30,2);

p10 = prctile(d,10,2);
p20 = prctile(d,20,2);
p30 = prctile(d,30,2);
p40 = prctile(d,40,2);
p50 = prctile(d,50,2);
p60 = prctile(d,60,2);
p70 = prctile(d,70,2);
p80 = prctile(d,80,2);
p90 = prctile(d,90,2);

p75 = prctile(d,75,2);
p95 = prctile(d,95,2);

IQR = iqr(d,2);

IQRd_diff = iqr(d_diff,2);


IQRdd_diff = iqr(dd_diff,2);

%%
IQRcorr = [];
M2r = [];
M3r = [];
M4r = [];
for i = 1:size(d,1)
    
xxx = d(i,:);
[r,lags]=xcorr(xxx,'coeff');

r=r(length(xxx):end);
lags = lags(length(xxx):end);

IQRcorr1 = iqr(r,2);

IQRcorr = [IQRcorr;IQRcorr1];

m2r = moment(r,2,2);
m3r = moment(r,3,2);
m4r = moment(r,4,2);

M2r1 = m2r.^(1/2);
M3r1 = abs(m3r).^(1/3);
M4r1 = m4r.^(1/4);


M2r = [M2r;M2r1];
M3r = [M3r;M3r1];
M4r = [M4r;M4r1];

end

X_FeatureSpace = [M2, M3, M4, IQR,  M3d_diff, M4d_diff, IQRd_diff, IQRdd_diff];
