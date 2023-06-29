clc;
clear;
dataPoints=8528;
inputSignalSize=9000;
data=zeros(dataPoints,inputSignalSize);
biggerThan30Seconds=zeros(1,dataPoints);
smallerThan30Seconds=zeros(1,dataPoints);

for index=1:dataPoints
    index
    path=strcat("all-data/A0",num2str(index,'%04.f'));
    tmp=load(path);
    len=length(tmp.val);
    if len>inputSignalSize
      tmp.val=tmp.val(1:inputSignalSize);
      len=inputSignalSize;
      biggerThan30Seconds(index)=1;
    end 
      if len<inputSignalSize
      smallerThan30Seconds(index)=1;
    end
   data(index,1:len)=tmp.val;
end