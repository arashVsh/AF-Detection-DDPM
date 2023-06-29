function [features] = BEAT_BEAT_TIMEFEATURES_ver03(x,sampleRate,QRS)
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

warning('off','all')

if length(QRS)>3
    
    [beat] = beat_beat_BLACKSWAN_ver2(x, QRS);
    
    if ~isempty(beat)
         cutOffFreq1 = 1.1; filterOrder1 = 2;
        [bbfilter, aafilter] = butter(filterOrder1, cutOffFreq1/(sampleRate/2),'high');
        
        for i = 1:size(beat,2)
            
            stack = beat{1, i}; stack = stack(:)';
            stack = smooth(filtfilt(bbfilter, aafilter, stack));
            stack = detrend(stack);
            stack = stack - mean(stack);
            stack = stack/std(stack);
            [feature1_1(i,:)] = Frequency_Beat_TIMEFEATURES_ver01(stack, sampleRate);
            %stack = (stack - min(stack)) / (max(stack) - min(stack));
            %stack = (stack * range2) + lim1;
            LOCR  = beat{2, i};
            indx1 = LOCR - (0.2*sampleRate);
            factor = 0.19;
            while  indx1<1
                indx1 = LOCR - floor((factor*sampleRate));
                factor = factor - 0.01;
            end
            clear factor
            [pks, locs] = max(stack(indx1:LOCR - floor(0.08*sampleRate)));
            indxP1 =  (indx1 + locs - 1  -  (0.06*sampleRate)); 
            factor = 0.05;
            while  indxP1<1
                indxP1 =  (indx1 + locs - 1  -  floor(factor*sampleRate) ); 
                factor = factor - 0.001;
            end
            clear factor
            indxP2 = (indx1 + locs - 1 + (0.06*sampleRate)); 
            factor = 0.05;
            while (indxP2>=LOCR)
                indxP2 = (indx1 + locs - 1 + (factor*sampleRate));
                factor = factor - 0.001;
            end
            clear factor
            %if indxP2>=LOCR
            %   indxP2 = (LOCR - floor(0.08*sampleRate));
            %end
            if indxP2<= (indxP1 + 2)
                indxP2 = indxP1 + floor((0.10*sampleRate));
            end
            Pamlitude(i) = pks; clear pks locs indx1
            indxQ1 = indxP2 + 2;
            [pks, locs] = min(stack(indxQ1:LOCR));
            %indxQ2 = indxQ1 + floor((LOCR - indxQ1) / 2);
            %indxQ2 = LOCR - floor((0.10*sampleRate)/2);
            indxQ2 = indxQ1 + locs + 2 ;
            factor = 2;
            while (indxQ2 <= indxQ1)
                indxQ2 = indxQ1 + floor((LOCR - indxQ1) / factor);
                factor = factor *2;
                if (indxQ2>= LOCR)
                    indxQ2 = indxQ2 - 1;
                    break
                end
            end
            Qamlitude(i)  = pks; clear pks locs indx1
            indxR1 = indxQ2 + 2; 
            factor = 1;
            while (indxR1 >= LOCR)
                indxR1 = (LOCR - factor);
                factor = factor + 1;
            end
             if (LOCR+floor(0.12*sampleRate)) < length(stack)
                [pks, locs] = min(stack(LOCR:LOCR+floor(0.12*sampleRate)));
            else
                factor = 0.12;
                while ((LOCR+floor(factor*sampleRate)) >= length(stack))
                    factor = factor - 0.01;
                end
                [pks, locs] = min(stack(LOCR:LOCR+floor(factor*sampleRate)));
                clear factor
            end
            indxS1 = LOCR + locs - 3;
            indxR2 = indxS1 - 2;
            if indxR2<=indxR1
                indxR2 = LOCR + floor((0.10*sampleRate)/2);
            end
            if indxS1<=indxR2
                indxS1 = indxR2 + 2; 
            end
            indxS2 = LOCR + locs - 1 + (0.06*sampleRate);
            if indxS2<=indxS1
                indxS2 = LOCR + (0.12*sampleRate); 
            end
            if indxS2>length(stack)
                indxS2 = length(stack);
            end
            Samlitude(i) = pks;  clear pks locs factor
            if indxS2 ~= length(stack)
                
                indxT1 = indxS2 +2 ;
                indxT2 = LOCR + (0.42*sampleRate);
                if indxT2>length(stack)
                    indxT2 = length(stack);
                end
                if indxT2> indxT1+2
                    Tamplitude(i) = max(stack(indxT1:indxT2));
                else
                    Tamplitude(i) = NaN;
                end
            else
                Tamplitude(i) = NaN;
            end
            XP = indxP1: indxP2;
            YP = stack(XP);   E1(i) = sum(YP.^2);
            XR = indxR1: indxR2;
            YR = stack(XR);  E2(i) = sum(YR.^2);
            XS = indxS1: indxS2;
            YS = stack(XS);  E3(i) = sum(YS.^2);
            if  (isnan(Tamplitude(i)) ==0)
                XT = indxT1: indxT2;
            else
                indxT2 = 1; indxT1 = 2;
            end
            if indxT2> indxT1+2
                YT = stack(XT);  E4(i) = sum(YT.^2);
            elseif   (indxT2<= indxT1+2) ||  (isnan(Tamplitude(i)) ==1)
                YT = NaN;
                E4(i) = NaN;
            end
            [~, I] = max(YP);
            tttemp =  YP(1:I);
            coefficients = polyfit(1:length(YP(1:I)), tttemp(:)', 1); clear tttemp
            a1(i) = coefficients(1);
            clear coefficients
            tttemp = YP(I:length(YP));
            coefficients = polyfit(1:length(YP(I:length(YP))), tttemp(:)', 1);  clear tttemp
            a2(i) = coefficients(1);
            clear coefficients I
            [~,I] = max(YR);
             tttemp = YR(1:I);
            coefficients = polyfit(1:length(YR(1:I)), tttemp(:)', 1);  clear tttemp
            a3(i) = coefficients(1);
            clear coefficients
            tttemp = YR(I:length(YR));
            coefficients = polyfit(1:length(YR(I:length(YR))),  tttemp(:)', 1);  clear tttemp
            a4(i) = coefficients(1);
            clear coefficients I
            if ~isnan(YT)
                [~,I] = max(YT);
                tttemp = YT(1:I);
                coefficients = polyfit(1:length(YT(1:I)), tttemp(:)', 1);  clear tttemp
                a5(i) = coefficients(1);
                clear coefficients
                tttemp = YT(I:end);
                coefficients = polyfit(1:length(YT(I:end)), tttemp(:)', 1);  clear tttemp
                a6(i) = coefficients(1);
                clear coefficients I
            else
                a5(i) = NaN;  a6(i) = NaN;
            end
            %%
            [~, I] = max(YP);
            tttemp = YP(I:end);
            coefficients = polyfit(1:length(YP(I:end)), tttemp(:)', 1);  clear tttemp
            AB = polyval(coefficients,linspace(1,(length(YP(I:end))),20)); clear coefficients I
            tttemp = stack(indxQ2:indxR1);
            coefficients = polyfit(1:length(stack(indxQ2:indxR1)),  tttemp(:)', 1);   clear tttemp
            BC = polyval(coefficients,linspace(1,length(stack(indxQ2:indxR1)),20)); clear coefficients 
            theta1(i) = abs( acosd( dot(AB,BC)/(norm(AB)*norm(BC)) ));
            tttemp = stack(indxR1:LOCR); 
            coefficients = polyfit(1:length(stack(indxR1:LOCR)), tttemp(:)', 1);  clear tttemp
            CD = polyval(coefficients,linspace(1,length(stack(indxR1:LOCR)),20)); clear coefficients 
            tttemp = stack(LOCR:indxR2); 
            coefficients = polyfit(1:length(stack(LOCR:indxR2)),tttemp(:)', 1); clear tttemp
            DE = polyval(coefficients,linspace(1,length(stack(LOCR:indxR2)),20)); clear coefficients 
            theta2(i) =  abs( acosd( dot(CD,DE)/(norm(CD)*norm(DE)) ));
            Ytemp = stack(LOCR:indxS2);
            [~, I] = min(Ytemp);
            tttemp = Ytemp(1:I); 
            coefficients = polyfit(1:length(Ytemp(1:I)), tttemp(:)', 1);  clear tttemp
            EF = polyval(coefficients,linspace(1,length(Ytemp(1:I)),20)); clear coefficients 
            tttemp = Ytemp(I:end); 
            coefficients = polyfit(1:length(Ytemp(I:end)), tttemp(:)', 1);   clear tttemp
            FG = polyval(coefficients,linspace(1,length(Ytemp(I:end)),20)); clear coefficients I
            theta3(i) =  abs( acosd( dot(EF,FG)/(norm(EF)*norm(FG)) ));
            % ------------------------------------------------------------------------------------------------------------------------
            %%
            
            distPR(i) = abs(indxR1 - indxP1);
            AMPPT(i) = abs(Tamplitude(i) - Pamlitude(i));
            RAMP(i) = stack(LOCR);
            distRT(i) = abs(indxT1 - LOCR);
            Pwave_inf1(i) = max(YP);
            
            clear YP YQ YR YS YT XP XQ XR XS XT Ytemp I  coefficients
            clear AB BC CD DE EF FG
            clear stack indxP1 indxP2 indxQ1 indxQ2 indxR1 indxR2 indxS1 indxS2 indxT1 indxT2 LOCR
        end
        
        features  = [nanmean(a2), nanmean(a5),...
                         nanstd(a6),...
                         nanmedian(a3), nanmedian(a6),...
                         iqr(a1), iqr(a6),...
                         prctile(a5,5), prctile(a5,90), prctile(a6,5),...
                         iqr(theta3),...
                         nanmedian(E4) ,...
                         (max(E1) - min(E1)),...
                         prctile(E1,5), prctile(E1,90), prctile(E4,5),...
                         nanmean(distPR), nanstd(distPR), iqr(distPR), prctile(distPR,5),...
                         nanmean(AMPPT), nanstd(AMPPT), iqr(AMPPT),...
                         nanmean(RAMP), nanstd(RAMP),...
                         nanmean(distRT), nanstd(distRT),...
                         nanstd(feature1_1,1),...
                         (max(diff(QRS))/sampleRate)-(min(diff(QRS))/sampleRate)];
                        
        
        clear a1 a2 a3 a4 a5 a6 theta1 theta2 theta3 E1 E2 E3 E4
    else
        features = NaN*ones(1, 34);
    end
    
else
    features = NaN*ones(1, 34);
end
