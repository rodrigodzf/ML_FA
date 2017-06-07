% Period = 100;
% NumPeriod = 3;
% Range = [-1 1];
% % Specify the frequency range of the signal. 
% % For a sum-of-sinusoids signal, you specify 
% % the lower and upper frequencies of the passband 
% % in fractions of the Nyquist frequency.
% % In this example, use the entire frequency 
% % range between 0 and Nyquist frequency.
% Band = [0 1];
% [u,freq] = idinput([Period 1 NumPeriod],'sine',Band,Range);
% %%
% 

%%
% Sine of 1 Period
Amp = 1;
Fs = 44100;              % Sampling frequency 
Sp = 1/Fs;               % Sampling period 
L = 1000;                % Length of signal (Optimized with mult of 2)
% length = 0.001;
% length_in_samples = Sp * 128;
T = (0:L-1) * Sp;
freq = 2000;
x = Amp * sin(2 * pi * freq * T);
% disp(size(x));
plot(T,x);
%% FFT

n = 2^nextpow2(L);
% dim = 2;

% fft, padded with zeros if X has less
%     than N points and truncated if it has more.
 
Y = fft(x);

P2 = abs(Y/L);                  % Get 2-sided spectrum
P1 = P2(1:L/2+1);               % Get one side
P1(2:end-1) = 2*P1(2:end-1);    % Mirror 

f = Fs*(0:(L/2))/L;
plot(f,P1)
title('Single-Sided Amplitude Spectrum of S(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% size(f)
% plot(Y);

%% Noise Signal
Amp = 1;
Fs = 44100;              % Sampling frequency 
Sp = 1/Fs;               % Sampling period 
L = 512;                 % Length of signal (Optimized with mult of 2)
T = (0:L-1) * Sp;
disp(size(T));
X = Amp * randn(size(T)); % Signal in a column vector


% Get FFT
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;



% subplot(P1,f);


% d = fdesign.lowpass('Fp,Fst,Ap,Ast',3,5,0.5,40,100);
% Hd = design(d,'equiripple');
% output = filter(Hd,input);

%   Design a lowpass FIR filter with order 350 and cutoff frequency
%   of 150 Hz. The sample rate is 1.5 KHz. Filter a long vector of 
%   data using the overlap-add method to increase speed.

% D = designfilt('lowpassfir', 'FilterOrder', 30, ...
%      'CutoffFrequency', 10000, 'SampleRate', Fs);

% Butterworth
fc = 10000; % Cutoff
fs = Fs;  % Samplerate
order = 10;
[b,a] = butter(order,fc/(Fs/2),'low');

% L = 512;
% data = randn(1,L);  
% plot(data);

% output = fftfilt(b,X);
output = filter(b,a,X);

% hold on

% Get FFT
Yout = fft(output);
P2out = abs(Yout/L);
P1out = P2out(1:L/2+1);
P1out(2:end-1) = 2*P1out(2:end-1);
% f = Fs*(0:(L/2))/L;


% figure('Name','Noise Signal')
signal1 = subplot(3,1,1); 
plot(signal1,T,X)
title(signal1, 'Signal');

signal2 = subplot(3,1,2); 
plot(signal2,f,P1)
title(signal2, 'FFT');

% figure('Name','Filtered Signal')
signal3 = subplot(3,1,3); 
plot(signal3,f, P1out)
title(signal3, 'Filtered');
% legend('Input Data','Filtered Data');


%% Noise Signal
Amp = 1;
Fs = 44100;              % Sampling frequency 
Sp = 1/Fs;               % Sampling period 
L = 512;                 % Length of signal (Optimized with mult of 2)
T = (0:L-1) * Sp;
% disp(size(T));
% X = Amp * randn(size(T)); % Signal in a column vector
f_ = Fs*(0:(L/2))/L;

X = sawtooth(2*pi*441*T);
% Change Filter params and store
% freqs = [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ];
freqs = 1000:100:10000;
% freqs = freqs * 10;
fftStore = zeros(size(freqs, 2), L/2+1);
for i = 1:length(freqs) % For Each freq
    
    f = freqs(i);
    % Butterworth
    fc = f; % Cutoff
    fs = Fs;  % Samplerate
    order = 10;
    [b,a] = butter(order,fc/(fs/2));
    
    % Filter signal
    output = filter(b,a,X);
    
    % Get fft and store
    Y = fft(output);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);

    fftStore(i,:) = P1;
end

figure
% plot(f_,fftStore);  
plot(fftStore);
% pred = csvread('/Users/rodrigodiaz/Development/ML_05_2017/keras/pred.csv');
% figure
% plot(f_,pred)

% prepend cutoff parameter
fftStoreExport = [freqs' fftStore];
%%
csvwrite('fftstore.csv', fftStoreExport);

%%

pred = csvread('/Users/rodrigodiaz/Development/ML_05_2017/keras/pred-loss-0.0058-epoch-20-batch-1-linear_dense.csv');
% figure
% plot(f_,pred)
% axis([0 20000 0 0.25])
% size(pred)
plot(pred)
%%

fftread = csvread('/Users/rodrigodiaz/Development/ML_05_2017/keras/fftstore.csv');
figure
plot(f_,fftread(:,2:end));  
axis([0 20000 0 0.25])

% for i=size(fftread,1)
%     
% end
fsec = fftread(:,end/2);
figure
plot(fftread(:,2:end))
% is the value linear?

%%

testread = csvread('/Users/rodrigodiaz/Development/ML_05_2017/keras/predsvm.csv');
plot(testread);
plot(f_,testread);

%% Download dataset
HOMEIMAGES = './Images';
HOMEANNOTATIONS = './Annotations';
folderlist = {'05june05_static_street_porter'};
% LMinstall (folderlist, HOMEIMAGES, HOMEANNOTATIONS);
% read the image and annotation struct:
% [annotation, img] = LMread(folderlist, HOMEIMAGES);
% filename = fullfile(HOMEANNOTATIONS, '05june05_static_street_porter', '05june05_static_street_porter.xml');
% % plot the annotations
% LMplot(annotation, img)
database = LMdatabase(HOMEANNOTATIONS);
[D,j] = LMquery(database, '05june05_static_street_porter', '05june05_static_indoor');
LMdbshowscenes(database(j), HOMEIMAGES); % this shows all the objects