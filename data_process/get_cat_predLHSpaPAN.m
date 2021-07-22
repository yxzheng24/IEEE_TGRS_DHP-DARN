clear;close all;
overlap = 1:71;
%% Load dataset
folder = './pavia_lhsconv';
folder2 = './DHP_SR/DHP_SR_results/mats_pavia';

filepaths = dir(fullfile(folder,'*.mat'));
filepaths2 = dir(fullfile(folder2,'*.mat'));

for i = 1 : length(filepaths)

image_s=load(fullfile(folder,filepaths(i).name));
I_REF =  image_s.HR;
%% Genertate HR-PAN 
PAN = mean(I_REF(:,:,overlap),3);

image_lhs=load(fullfile(folder2,filepaths2(i).name));
I_paLHSpred = double(image_lhs.pred);
%% Genertate the residual HSI
I_respa = I_REF - I_paLHSpred;

%% Concatenation upsampled HSI and HR-PAN along the spectral dimension
predLHSpa_Pan = cat(3,I_paLHSpred,PAN);

fname = strcat('Res_',filepaths(i).name);
save(['./Respavia_lhsconv/',fname1],'I_respa');

fname2 = strcat('cat_PAN_',filepaths(i).name);
save(['./cat_LHS_PAN/I_predLHSpa_Pan/',fname2],'predLHSpa_Pan');

end