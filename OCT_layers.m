I=imread('Default_0003_Mode2D.jpg');
Igray=rgb2gray(I);
figure();imshow(I);title('OCT image of tape');

%Taking FFT of image
Ifd = fft2(Igray);
% figure();imagesc(log(abs(fftshift(Ifd))));title('Magnitude of OCT Tape');
% figure();imagesc(angle(Ifd));title('Angle of OCT Tape');

%Edge detection without preprocessing
prewitt_filter1=fspecial('prewitt');
prewittI1=imfilter(Igray,prewitt_filter1*2);
figure();imshow(prewittI1);title('Edge Detection');

% Min filter 1 to remove the salt noise
Imin=ordfilt2(prewittI1,4,ones(3,3));
figure();imshow(Imin);title('Min Filter');

% Image Closing to join each layer
SE = strel('line', 8, 0);
Idilate=imclose(Imin,SE);
figure();imshow(Idilate);title('Closing');

% Image Binarization
BW = imbinarize(Idilate);
figure();imshow(BW);title('Binarization');
