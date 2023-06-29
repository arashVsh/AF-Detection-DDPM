
function [feature] = NEWPS_BLACKSWAN(x, sampleRate, QRS)
%%
% INPUT:
% x
% sampleRate
% QRS
%
% OUTPUT:
% features
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
rng(0)
if length(QRS)>3
    
    [beat] = beat_beat_BLACKSWAN(x, QRS);
    
    for i = 1:length(beat)
        
        xx = beat{i};
        xx = xx - mean(xx);
        %%
        [T, T1] = phasespace(xx, 2, 3);
        T = T./(size(T,1));
        T(:,1) = (T(:,1)-min(T(:,1)))/(max(T(:,1))-min(T(:,1)));
        T(:,2) = (T(:,2)-min(T(:,2)))/(max(T(:,2))-min(T(:,2)));
        clear fittedY fittedX; fittedY = linspace(min(T(:)), max(T(:)), T1); fittedX = 0.5*ones(1,length(fittedY));
         %[xi,yi] = polyxpoly(T(:,1),T(:,2),fittedX,fittedY);
        P = InterX([T(:,1)';T(:,2)'],[fittedX;fittedY]);
        xi = P(1,:);
        yi = P(2,:);
        clear P
        %%
        if ~isempty(yi)
            [~, I1] = max(yi); [~, I2] = min(yi);
            f1(1,i) = sqrt((yi(I1) - yi(I2) ).^2);
            clear I1 I2
        else
            f1(1,i)  = NaN;
        end
        %----------------------------------------------------------------------------------------------------------------------------
        clear xx xi yi T T1
    end
    
    % ------------------------------------------------------------------------------------------------------------------------------
    XXYY = diff(QRS')/sampleRate;
    XX1 = XXYY;
    XX2 = (mean(XXYY) - XXYY).^2;
    f = fit(XX1,XX2,'poly2');
    % ------------------------------------------------------------------------------------------------------------------------------
    Ax = min(XXYY);                                        Ay =  abs(mean(XXYY) - min(XXYY));
    Bx = max(XXYY);                                        By  = abs(mean(XXYY) - max(XXYY));
    [Cy, I] = min(abs( mean(XXYY) -  XXYY )); Cy = Cy(1); I = I(1);
    Cx = XXYY;  Cx = Cx(I);                   
    f1 = (By - Ay) / (Bx - Ax);
    a = sqrt( ((Bx - Cx ).^2) + ((By - Cy).^2)  ); b = sqrt( ((Ax - Cx ).^2) + ((Ay - Cy).^2)  ); c = sqrt( ((Bx - Ax ).^2) + ((By - Ay).^2)  );
    f5 = a + b + c;
    f6 = 0.5*(((Ax - Cx)*(By - Ay)) - ((Ax - Bx)*(Cy - Ay)));
    f7 = (4*sqrt(3)*f6) / ( (a^2) + (b^2) + (c^2) );
    % ------------------------------------------------------------------------------------------------------------------------------
    XX11 = XXYY(1:end-1);
    YY22 = XXYY(2:end);
    DISTI = (YY22 - XX11) ./ (sqrt(2));
    
    indxU = find(DISTI ==0);
    indxO = find(DISTI >0);
    indxD = find(DISTI <0);
    
    f8 = length(indxU);
    f9 = length(indxO);
    f10 = length(indxD);

    UU = 0; UO = 0; UD = 0; OU = 0; OO = 0; OD  = 0;DU = 0; DO = 0; DD = 0;
    for Pi =1:length(YY22)-1
        
        if ~isempty(find(Pi==indxU)) && ~isempty(find((Pi+1)==indxU))
            UU = UU +1;
        end
        if ~isempty(find(Pi==indxU)) && ~isempty(find((Pi+1)==indxO))
            UO = UO +1;
        end
        if ~isempty(find(Pi==indxU)) && ~isempty(find((Pi+1)==indxD))
            UD = UD +1;
        end
        if ~isempty(find(Pi==indxO)) && ~isempty(find((Pi+1)==indxU))
            OU = OU +1;
        end
        if ~isempty(find(Pi==indxO)) && ~isempty(find((Pi+1)==indxO))
            OO = OO +1;
        end
        if ~isempty(find(Pi==indxO)) && ~isempty(find((Pi+1)==indxD))
            OD = OD +1;
        end
        if ~isempty(find(Pi==indxD)) && ~isempty(find((Pi+1)==indxU))
            DU = DU +1;
        end
        if ~isempty(find(Pi==indxD)) && ~isempty(find((Pi+1)==indxO))
            DO = DO +1;
        end
        if ~isempty(find(Pi==indxD)) && ~isempty(find((Pi+1)==indxD))
            DD = DD +1;
        end

    end
    % ------------------------------------------------------------------------------------------------------------------------------
    feature = [nanmean(f1), f.p2, f5, f6, OU, OO, OD, DU, DD];
else
    feature = NaN*ones(1,9);
end

