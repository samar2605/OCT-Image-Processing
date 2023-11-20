% Reading the image2
I2 = imread("OCT/Default_0014_Mode2D.jpg");

% Converting image to grayscale
OCT_image2 = rgb2gray(I2);
figure('Name','Original Image');
imshow(OCT_image2);
title('Original Image');
imwrite(OCT_image2,'Tape/oct_image.jpg')

%%
% plotting the pdf of OCT image
[counts,binLocations] = imhist(OCT_image2);
figure('Name','PDF of OCT image');
imhist(OCT_image2);
title('PDF of Original Image');
imwrite(imhist(OCT_image2),'Tape/pdf_oct_image.jpg')


%% speckle correction
%apply median filter on the image2
img_med= medfilt2(OCT_image2,[5 5]);
figure('Name','Image after median filtering');
imshow(img_med);
title('Image after median filtering');

img_weiner = wiener2(img_med,[11 11]);
figure('Name','Image after wiener filtering');
imshow(img_weiner);
title('Image after wiener filtering');

img_bilat= imbilatfilt(img_weiner,9);
figure('Name','Image after bilateral smoothening');
imshow(img_weiner);
title('Image after bilateral smoothening');

img_gama= imadjust(img_bilat,[],[],1.3);
figure('Name','Image after gamma correction');
imshow(img_gama)
title('Image after gamma correction');

img_speckle = (OCT_image2 - img_gama);
figure('Name','Image of the speckle');
imshow(img_speckle)
title('Image of the speckle');

figure('Name','Original vs corrected image');
imshowpair(OCT_image2,img_gama, 'montage');
title('Original (Left) vs Speckle corrected (Right) image');
imwrite(img_gama,'Tape/speckles_corrected_median.jpg')
imwrite(cat(1,OCT_image2,img_gama),'Tape/speckles_corrected_median_montage.jpg')

%% Frequency Distribution Analysis
% frequency distribution analysis of original image
% Performing 2D Fourier Transform
fft_image = fft2(double(OCT_image2));

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
fft_image = fft2(double(img_gama));

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
%% plotting the histograms of original image, wiener filtered image, final smoothened image

figure('Name','PDF of OCT image');
imhist(OCT_image2);
title('PDF of Original Image 2');

figure('Name','PDF of  wiener filtered image');
imhist(img_wiener);
title('PDF of wiener filtered Image');

figure('Name','PDF of bilateral smoothened image');
imhist(img_bilat);
title('PDF of bilateral filtered image');

