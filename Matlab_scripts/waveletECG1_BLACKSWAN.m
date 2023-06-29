function X_FeaturesSpace = waveletECG1_BLACKSWAN(X)

%%
% INPUT:
% X
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
    level = 7;
    Wtype = 'db4';

    [c,l] = wavedec(x,level,Wtype);
    app = appcoef(c,l,Wtype,level);

    cD7 = detcoef(c,l,level);
    cD6 = detcoef(c,l,level-1);
    cD5 = detcoef(c,l,level-2);
    cD4 = detcoef(c,l,level-3);
    cD3 = detcoef(c,l,level-4); 
    cD2 = detcoef(c,l,level-5);
    cD1 = detcoef(c,l,level-6);

    recD7 = upcoef('d',cD7,Wtype,level,l(end));
    recD6 = upcoef('d',cD6,Wtype,level-1,l(end));
    recD5 = upcoef('d',cD5,Wtype,level-2,l(end));
    recD4 = upcoef('d',cD4,Wtype,level-3,l(end));
    recD3 = upcoef('d',cD3,Wtype,level-4,l(end));

    Mapp = zeros(size(app));
    McD2 = zeros(size(cD2));
    McD1 = zeros(size(cD1));

    Mc = [Mapp,cD7,cD6,cD5,cD4,cD3,McD2,McD1];
    recD = waverec(Mc,l,Wtype);
    % ------------------------------------------------------------------------------------------------------------------------------
    c_IQR = [iqr(cD7), iqr(cD6), iqr(cD5), iqr(cD4), iqr(cD3)];
    % ------------------------------------------------------------------------------------------------------------------------------
    c_var = [var(cD7), var(cD6), var(cD5), var(cD4),var(cD3)];
    % ------------------------------------------------------------------------------------------------------------------------------
    percent = 25;
    c_prctile = [prctile(cD7,percent), prctile(cD6,percent), prctile(cD5,percent), prctile(cD4,percent),prctile(cD3,percent)]; 
    % ------------------------------------------------------------------------------------------------------------------------------
    %  AR model
    data = recD;
    order = 4;
    [ar_coeffs,NoiseVariance] = arburg(data,order);
    % ------------------------------------------------------------------------------------------------------------------------------
    %ACF
    [r,lags] = xcorr(recD,'coeff');
    r = r(length(recD):end);
    lags = lags(length(recD):end); %#ok<NASGU>
    minTH = 0.2;
    [pks, locs] = findpeaks(r,'MINPEAKHEIGHT',minTH); %#ok<ASGLU>

    numberLargePeak = length(pks);
    [pks,locs] = findpeaks(r); %#ok<ASGLU>
    
    if  length(diff(locs)) == 0 %#ok<ISMT>
        c_ACF(1,1) = numberLargePeak;
    else
        c_ACF = [numberLargePeak]; %#ok<NBRAK>
    end
    % ------------------------------------------------------------------------------------------------------------------------------
    W_features1 = [c_IQR, c_var, c_prctile, NoiseVariance, ar_coeffs(2:end),c_ACF/(length(X)/300)];
    % ------------------------------------------------------------------------------------------------------------------------------

    S_features1 = statECG1_BLACKSWAN(recD);

    M_features1 = morphoECG1_BLACKSWAN(recD)./(length(X)/300);

    features1 = [W_features1,S_features1,M_features1];


    features = [features; features1]; %#ok<AGROW>

end

X_FeaturesSpace = features;