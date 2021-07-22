clear;close all;

%% Load dataset
folder = './pavia_subimgs';

%% Settings
ratio = 4;
size_kernel=[8 8];
sig = (1/(2*(2.7725887)/ratio^2))^0.5;
start_pos(1)=1; 
start_pos(2)=1;
count = 0;

filepaths = dir(fullfile(folder,'*.mat'));

for i = 1 : length(filepaths)
image_s=load(fullfile(folder,filepaths(i).name));
I_REF = image_s.subim;
%% Normalization
A=max(max(max(I_REF)));
B=min(min(min(I_REF)));
HR=(I_REF-B)./(A-B);
%% Genertate LR-HSI, bic-HSI, nea-HSI 
[LR,KerBlu]=conv_downsample(HR,ratio,size_kernel,sig,start_pos);

bicubic = imresize(LR, 4, 'bicubic');
nearest = imresize(LR, 4, 'nearest');

fname = strcat('lhsconv_',filepaths(i).name);
save(['./pavia_lhsconv/',fname],'HR','LR','bicubic','nearest','-v7');

count=count+1;
 end
