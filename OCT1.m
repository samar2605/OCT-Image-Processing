% Reading the image
I = imread("OCT/Default_0001_Mode2D.jpg");
I2 = imread("OCT/Default_0014_Mode2D.jpg");

% Converting image to grayscale
OCT_image = rgb2gray(I);
OCT_image2 = rgb2gray(I2);
figure('Name','Original Image');
imshow(OCT_image);
title('Original Image');

imshow(OCT_image2);
title("Original Image 2");

imwrite(OCT_image,'Tape/oct_image.jpg')

%%
% plotting the pdf of OCT image
[counts,binLocations] = imhist(OCT_image);
figure('Name','PDF of OCT image');
imhist(OCT_image);
title('PDF of Original Image');

imhist(OCT_image2);
title('PDF of Original Image2');
imwrite(imhist(OCT_image),'Tape/pdf_oct_image.jpg')

%%
%speckle detection using thresholding
threshold = 50; 
speckle_mask = OCT_image < threshold;
figure('Name','Speckle');
imshow(speckle_mask);
title('Speckle');
imwrite(speckle_mask,'Tape/speckles.jpg')
%%
%speckle correction median filter
kernel_size = 5;
corrected_image = medfilt2(OCT_image, [kernel_size, kernel_size]); 
figure('Name','Image after Speckle correction');
imshow(corrected_image);
title('Image after Speckle correction');

figure('Name','Original vs corrected image');
imshowpair(OCT_image,corrected_image, 'montage');
title('Original (Left) vs Speckle corrected (Right) image');
imwrite(corrected_image,'Tape/speckles_corrected_median.jpg')
imwrite(cat(1,OCT_image,corrected_image),'Tape/speckles_corrected_median_montage.jpg')
%%
% Speckle correction using gaussian filter
sigma = 0.9;
I_gaussian_3_3=imgaussfilt(OCT_image,sigma,'FilterSize',[3 3]);
I_gaussian_9_9=imgaussfilt(OCT_image,sigma,'FilterSize',[9 9]);
I_gaussian_27_27=imgaussfilt(OCT_image,sigma,'FilterSize',[27 27]);


figure('Name','Original vs corrected image');
imshowpair(OCT_image,I_gaussian_3_3, 'montage');
title('Original (Left) vs Speckle corrected (Right) image');
imwrite(I_gaussian_3_3,'Tape/speckles_corrected_gaussian3.jpg')

figure('Name','Original vs corrected image');
imshowpair(OCT_image,I_gaussian_9_9, 'montage');
title('Original (Left) vs Speckle corrected (Right) image');
imwrite(I_gaussian_9_9,'Tape/speckles_corrected_gaussian9.jpg')

figure('Name','Original vs corrected image');
imshowpair(OCT_image,I_gaussian_27_27, 'montage');
title('Original (Left) vs Speckle corrected (Right) image');
imwrite(I_gaussian_27_27,'Tape/speckles_corrected_gaussian27.jpg')

figure('Name','Gaussian filter')
montage(cat(2,I_gaussian_3_3,I_gaussian_9_9,I_gaussian_27_27));
title('Left: 3×3 Gaussian Filter, Middle: 9×9 Gaussian Filter,Right: 27×27 Gaussian Filter');
imwrite(cat(2,I_gaussian_3_3,I_gaussian_9_9,I_gaussian_27_27),'Tape/speckles_corrected_gaussian_montage.jpg')

%%
%speckle correction of Image2
% first we apply median filter
img_med= medfilt2(OCT_image2,[5 5]);
figure('Name','Image after applying median filter');
imshow(img_med);
title('Image after applying median filter');

% then we apply wiener filter on the above image
img_weiner = wiener2(img_med,[11 11]);
figure('Name','Image after applying wiener filter');
imshow(img_weiner);
title('Image after applying wiener filter');

% then applying bilateral filter on the above image
img_bilat= imbilatfilt(img_weiner,9);
figure('Name','Image after applying bilateral filter');
imshow(img_weiner);
title('Image after applying bilateral filter');

% then applying gamma correction on the above image
img_gama= imadjust(img_bilat,[],[],1.3);
figure('Name','Image after applying gamma correction');
imshow(img_gama);
title('Image after applying gamma correction');

%% marking speckle for the image2
% subtracting the cleaned image from the original gives the speckle
img_speck = (OCT_image2 - img_gama); 
imshow(img_speck)
title('Speckle in the original Image')

%%  Segmentation - method 1
% K-means clustering based image segmentation for corrected image
[L,Centers] = imsegkmeans(corrected_image,3);
segmented_image = labeloverlay(corrected_image,L);
figure('Name','Segmented Image');
imshow(segmented_image);
title("Segmented Image");

%% 
% Measure thickness of each layer
thickness_values = zeros(1, 3);

% Create a binary mask for each segment
segment_masks = cell(1, 3);

for i = 1:3
    segment_masks{i} = L == i;
end

for i = 1:3
    % Compute the thickness as the number of pixels along the vertical direction
    thickness_values(i) = sum(segment_masks{i}, 'all') / size(segment_masks{i}, 1);
end

% thickness values
disp('Thickness values for each layer(in pixels):');
disp(thickness_values);
%%
% layer segmentation
edges = edge(corrected_image, 'Canny');
se = strel('disk', 3);
dilated_edges = imdilate(edges, se);
close_edges = imclose(dilated_edges, se);
segmented_layers = imfill(close_edges, 'holes');
% segmented_layers = close_edges;
figure('Name','Image after Segmentation');
imshow(segmented_layers);
title('Image after Segmentation');

figure('Name','Original vs Segmented image');
imshowpair(OCT_image,segmented_layers, 'montage');
title('Original (Left) vs Segmented (Right) image');
 
%% Segmentation - method 2
% Apply morphological opening to enhance bright structures
se_opening = strel('disk',3);
opened_image = imopen(corrected_image, se_opening);

% Apply morphological closing to fill gaps in structures
se_closing = strel('disk',71);
closed_image = imclose(opened_image, se_closing);

% Compute the gradient to highlight edges
morph_gradient = imsubtract(closed_image, opened_image);

% Threshold the gradient image to obtain a binary segmentation
threshold = graythresh(morph_gradient);
binary_segmentation = imbinarize(morph_gradient, threshold);

figure('Name','Original vs Segmented image');
imshowpair(OCT_image,binary_segmentation, 'montage');
title('Original (Left) vs Segmented (Right) image');

%% Frequency Distribution Analysis
% frequency distribution analysis of original image
% Performing 2D Fourier Transform
fft_image = fft2(double(OCT_image));

% Shifting zero-frequency components to the center
fft_image_shifted = fftshift(fft_image);

% magnitude spectrum (log scale for visualization)
magnitude_spectrum = log(abs(fft_image_shifted) + 1);

figure('Name','Magnitude'); 
imagesc(magnitude_spectrum);
title('Magnitude spectrum of original image');

%phase spectrum
phase_spectrum = angle(fft_image_shifted);
figure('Name','Phase'); imagesc(phase_spectrum);
title('Phase spectrum of original image')

%%
% frequency distribution analysis of corrected image
% Performing 2D Fourier Transform
fft_image = fft2(double(corrected_image));

% Shifting zero-frequency components to the center
fft_image_shifted = fftshift(fft_image);

% magnitude spectrum (log scale for visualization)
magnitude_spectrum = log(abs(fft_image_shifted) + 1);

figure('Name','Magnitude'); 
imagesc(magnitude_spectrum);
title('Magnitude spectrum of corrected image');

%phase spectrum
phase_spectrum = angle(fft_image_shifted);
figure('Name','Phase'); imagesc(phase_spectrum);
title('Phase spectrum of corrected image')
