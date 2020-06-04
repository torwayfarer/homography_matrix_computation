
%%% read images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first pair of images
image_1=imread('image/H1_ex1.png');
image_2=imread('image/H1_ex2.png');
choose_detector=1;


% second pair of images
% uncomment everything below to choose the second pairs of images (it uses SURF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image_1=imread('image/H2_ex1.png');
%image_2=imread('image/H2_ex2.png');
%choose_detector=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% my own pictures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%image_1=imread('image/my_pic1.jpg');
%image_2=imread('image/my_pic2.jpg');
%choose_detector=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% process the images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make images grayscale
image_1=rgb2gray(image_1);
image_2=rgb2gray(image_2);

%%% resize the images
image_1 = imresize(image_1, 0.75);
image_2 = imresize(image_2, 0.75);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute interest points in each image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect corners using FAST algorithm and return cornerPoints object
% very good for 1st pair of images
if (choose_detector==1)
    disp('1st pair of images')
    points1_detected = detectFASTFeatures(image_1);
    points2_detected = detectFASTFeatures(image_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect SURF features and return SURFPoints object
% very good for 2nd pair of images
else
    points1_detected = detectSURFFeatures(image_1, 'MetricThreshold', 1000);
    points2_detected = detectSURFFeatures(image_2, 'MetricThreshold', 1000);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect corners using Harris–Stephens algorithm and return cornerPoints object
%points1_detected = detectHarrisFeatures(image_1);
%points2_detected = detectHarrisFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect BRISK features and return BRISKPoints object
%points1_detected = detectBRISKFeatures(image_1);
%points2_detected = detectBRISKFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%points1_detected = detectMinEigenFeatures(image_1);
%points2_detected = detectMinEigenFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%points1_detected = detectMSERFeatures(image_1);
%points2_detected = detectMSERFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect and store ORB keypoints
%points1_detected = detectORBFeatures(image_1);
%points2_detected = detectORBFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Detect KAZE features
%points1_detected = detectKAZEFeatures(image_1);
%points2_detected = detectKAZEFeatures(image_2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
title('interest points 1st image')
imshow(image_1)
hold on
plot(points1_detected);

figure
title('interest points 2nd image')
imshow(image_2)
hold on
plot(points2_detected);


% returns extracted feature vectors (descriptors) and their corresponding locations
[features1,valid_points1] = extractFeatures(image_1,points1_detected);
[features2,valid_points2] = extractFeatures(image_2,points2_detected);

% match features
% returns indices of the matching features in the two input feature sets
indices = matchFeatures(features1,features2);

% get the matched points in 2 interest points sets 
matchedPoints1 = valid_points1(indices(:,1),:);
matchedPoints2 = valid_points2(indices(:,2),:);

% visualize the matched points
figure
showMatchedFeatures(image_1,image_2,matchedPoints1,matchedPoints2, 'montage')
title('matched points');

figure 
showMatchedFeatures(image_1, image_2, matchedPoints1,matchedPoints2)
title('matched point (another representation)');


% set the parameters and constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of matched points 
number_of_corresp = matchedPoints1.Count;
% number of times to estimate homography
N=500;
% threshold distance
T_DIST=60;
% initial maximum number of inliers
MAX_inlier=-1;
% initial minimum standard deviation of distances
MIN_std=10e5;
p=0.99;
% number of points to estimate the homography
number_of_points=4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cycle of estimation starts
for iter_1=drange(1:N)
    
    % randomly choose 4 correspondences
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    rng('shuffle');
    random_numbers = randi([1, number_of_corresp], 1, number_of_points);
    points1_locations=matchedPoints1(random_numbers).Location;
    points2_locations=matchedPoints2(random_numbers).Location;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % check whether these points are colinear, if so, redo the above step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while (if_collinear(points1_locations)) || (if_collinear(points2_locations))
        rng('shuffle');
        random_numbers = randi([1, number_of_corresp], 1, number_of_points);
        points1_locations=matchedPoints1(random_numbers).Location;
        points2_locations=matchedPoints2(random_numbers).Location;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % normalize the points
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sim_matrix_1=similarity(points1_locations);
    sim_matrix_2=similarity(points2_locations);

    points1_locations_new=zeros(3, 4);
    points2_locations_new=zeros(3, 4);
    for i = 1:number_of_points
        points1_locations_new(:, i)=sim_matrix_1*[points1_locations(i, 1); points1_locations(i, 2); 1];
        points2_locations_new(:, i)=sim_matrix_2*[points2_locations(i, 1); points2_locations(i, 2); 1];
        
    end
    
    points1_locations_new=transpose(points1_locations_new);
    points2_locations_new=transpose(points2_locations_new);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % make an equation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    equation=zeros(number_of_points*2, 9);
    
    for iter_2=drange(1:number_of_points)
        
        equation(iter_2*2 - 1, :)=[points1_locations_new(iter_2, 1), points1_locations_new(iter_2, 2), 1, 0, 0, 0, ...
            -(points1_locations_new(iter_2, 1)*points2_locations_new(iter_2, 1)), -(points2_locations_new(iter_2, 1)*points1_locations_new(iter_2, 2)), ...
            -points2_locations_new(iter_2, 1)];
        equation(iter_2*2, :)=[0, 0, 0, points1_locations_new(iter_2, 1), points1_locations_new(iter_2, 2), 1, ... 
            -(points1_locations_new(iter_2, 1)*points2_locations_new(iter_2, 2)), -(points2_locations_new(iter_2, 2)*points1_locations_new(iter_2, 2)), ...
            -points2_locations_new(iter_2, 2)];
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % solve the equation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    solution=null(equation);
    solution=reshape(solution, [3, 3]);
    %solution=transpose(solution);
    DLT_norm=(sim_matrix_2\solution)*sim_matrix_1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % for each putative correspondence, calculate distance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    distances=zeros(1, number_of_corresp);
    inliers_indexes=zeros(1, number_of_corresp);
    for i=1:number_of_corresp
        number_of_inliers=0;
        coordinates_initial_1=matchedPoints1(i).Location;
        coordinates_initial_1(3)=1;
        coordinates_initial_1=transpose(coordinates_initial_1);
        
        coordinates_initial_2=matchedPoints2(i).Location;
        coordinates_initial_2(3)=1;
        coordinates_initial_2=transpose(coordinates_initial_2);
        
        
        coordinates_mimic_2=DLT_norm*coordinates_initial_1;
        coordinates_mimic_1=DLT_norm\coordinates_initial_2;
        
        curr_distance=sqrt((coordinates_initial_1(1, 1)-coordinates_mimic_1(1,1)/coordinates_mimic_1(3, 1))^2 ...
            +(coordinates_initial_1(2, 1)-coordinates_mimic_1(2,1)/coordinates_mimic_1(3, 1))^2)+...
                sqrt((coordinates_initial_2(1, 1)-coordinates_mimic_2(1,1)/coordinates_mimic_2(3, 1))^2 ...
            +(coordinates_initial_2(2, 1)-coordinates_mimic_2(2,1)/coordinates_mimic_2(3, 1))^2);
        
        distances(1, i)=curr_distance;
        
        % count the number of inliers m which has the distance <T_DIST
        if (curr_distance<T_DIST)
            number_of_inliers=number_of_inliers+1;
            inliers_indexes(1, i)=i;
        end

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % compute the standard deviation of the inlier distance curr_std
    curr_std=std(distances);
    
    % if(m > MAX inlier or (m == MAX_inlier and curr_std < MIN std))
    % update best H = Hcurr and record all the inliers
    if (number_of_inliers>MAX_inlier) || (number_of_inliers==MAX_inlier && curr_std<MIN_std)
        MAX_inlier=number_of_inliers;
        MIN_std=curr_std;
        best_homography=DLT_norm;
        best_inliers=zeros(1, number_of_inliers);
        k=1;
        for j=1:size(inliers_indexes, 2)
            if ~inliers_indexes(1, j)==0
                best_inliers(1, k)=inliers_indexes(1, j);
                k=k+1;
            end
        end
        
    end
    
end
% end of for loop

figure 
showMatchedFeatures(image_1, image_2, matchedPoints1(best_inliers).Location, matchedPoints2(best_inliers).Location)
title('inliers after RANSAC (before final estimation of H)');

%outliers_after_ransac=zeros(1, number_of_corresp-num_inliers);
outliers=[];
for i=1:number_of_corresp
    if ismember(i, best_inliers)
        continue;
    else
        outliers=[outliers, i];
    end
        
end

figure 
showMatchedFeatures(image_1, image_2, matchedPoints1(outliers).Location, matchedPoints2(outliers).Location)
title('outliers after RANSAC (before final estimation of H)');

% record the answer after for loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
answer_after_for_loop=transpose(best_homography/best_homography(3, 3));
disp('answer_after_for_loop: ');
disp(answer_after_for_loop);


disp('total number of correspondancies before guided matching ');
disp(number_of_corresp);


num_inliers=size(best_inliers, 2);
disp('number of inliers after RANSAC (before final computation of H): ');
disp(num_inliers);

disp('number of outliers after RANSAC (before final computation of H): ');
disp(size(outliers, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Transform 1st image into 2nd
% to get the reconstructed scene image, compare to the original scene image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tform = projective2d(answer_after_for_loop);
image_2_mimic = imwarp(image_1, tform, 'nearest', 'outputView', imref2d(size(image_1)));
matched_2 = imfuse(image_2_mimic, image_2,'Scaling','joint','ColorChannels',[1 2 0]);
figure
imshow(matched_2)
title('1st image transformed into 2nd based on 4 points');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Refinement: re-estimate H from all the inliers using the DLT algorithm.
% make the equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
equation=zeros(num_inliers*2, 9);
points1_locations=matchedPoints1(best_inliers).Location;
points2_locations=matchedPoints2(best_inliers).Location;   

for iter_2=drange(1:num_inliers)

    equation(iter_2*2 - 1, :)=[points1_locations(iter_2, 1), points1_locations(iter_2, 2), 1, 0, 0, 0, ...
        -(points1_locations(iter_2, 1)*points2_locations(iter_2, 1)), -(points2_locations(iter_2, 1)*points1_locations(iter_2, 2)), ...
        -points2_locations(iter_2, 1)];
    equation(iter_2*2, :)=[0, 0, 0, points1_locations(iter_2, 1), points1_locations(iter_2, 2), 1, ... 
        -(points1_locations(iter_2, 1)*points2_locations(iter_2, 2)), -(points2_locations(iter_2, 2)*points1_locations(iter_2, 2)), ...
        -points2_locations(iter_2, 2)];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% solve the equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if num_inliers==4
    solution=null(equation);
else
    [a, b, V] = svd(equation);
    solution=V(:, 9);
end

solution=reshape(solution, [3, 3]);
final_homography=solution*(1.0000/solution(3, 3));
if final_homography(3, 3)<0
    final_homography=final_homography*(-1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% determine final number of inliers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for_inliers=transpose(final_homography);
number_of_inliers_final=0;
inliers_indexes=zeros(1, number_of_corresp);
for i=1:number_of_corresp
    number_of_inliers=0;
    coordinates_initial_1=matchedPoints1(i).Location;
    coordinates_initial_1(3)=1;
    coordinates_initial_1=transpose(coordinates_initial_1);

    coordinates_initial_2=matchedPoints2(i).Location;
    coordinates_initial_2(3)=1;
    coordinates_initial_2=transpose(coordinates_initial_2);


    coordinates_mimic_2=for_inliers*coordinates_initial_1;
    coordinates_mimic_1=for_inliers\coordinates_initial_2;

    curr_distance=sqrt((coordinates_initial_1(1, 1)-coordinates_mimic_1(1,1)/coordinates_mimic_1(3, 1))^2 ...
        +(coordinates_initial_1(2, 1)-coordinates_mimic_1(2,1)/coordinates_mimic_1(3, 1))^2)+...
            sqrt((coordinates_initial_2(1, 1)-coordinates_mimic_2(1,1)/coordinates_mimic_2(3, 1))^2 ...
        +(coordinates_initial_2(2, 1)-coordinates_mimic_2(2,1)/coordinates_mimic_2(3, 1))^2);


    if (curr_distance<T_DIST)
        number_of_inliers_final=number_of_inliers_final+1;
        inliers_indexes(1, i)=i;
    end

end

final_inliers=zeros(1, number_of_inliers_final);
k=1;
for j=1:size(inliers_indexes, 2)
    if ~inliers_indexes(1, j)==0
        final_inliers(1, k)=inliers_indexes(1, j);
        k=k+1;
    end
end

disp('number of inliers for final H:');
disp(number_of_inliers_final);

figure 
showMatchedFeatures(image_1, image_2, matchedPoints1(final_inliers).Location, matchedPoints2(final_inliers).Location)
title('Final inliers');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% record the built-in answer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[builtin_homography, points1_builtin, points2_builtin] = estimateGeometricTransform(matchedPoints1, matchedPoints2, 'projective', 'MaxDistance', 10.0);
builtin_homography = builtin_homography.T;
disp('built-in answer:')
disp(builtin_homography);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% record the final answer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%my_answer=transpose(final_homography);
my_answer=final_homography;
disp('my answer:')
disp(my_answer);
final_homography=double(final_homography);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Transform 2nd image into 1st
% to get the reconstructed scene image, compare to the original scene image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tform_inverse = projective2d(inv(final_homography));
image_1_mimic = imwarp(image_2, tform_inverse, 'outputView', imref2d(size(image_2)));
matched_2 = imfuse(image_1_mimic, image_1,'Scaling','joint','ColorChannels',[1 2 0]);
figure
imshow(matched_2)
title('2nd image transformed into 1st');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transform 1st image into 2nd
% to get the reconstructed scene image, compare to the original scene image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tform = projective2d(final_homography);
image_2_mimic = imwarp(image_1, tform, 'outputView', imref2d(size(image_1)));
matched_2 = imfuse(image_2_mimic, image_2,'Scaling','joint','ColorChannels',[1 2 0]);
figure
imshow(matched_2)
title('1st image transformed into 2nd');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% compute guided matching
%{
guided_matching_1=[];
guided_matching_2=[];
threshold_for_matching=20;
k=1;
inliers_indexes=zeros(1, number_of_corresp);
for i=1:points1_detected.Count
    disp(i);
    minimum=50000;
    minimum_index=5;
    coordinates_initial_1=points1_detected(i).Location;
    coordinates_initial_1(3)=1;
    coordinates_initial_1=transpose(coordinates_initial_1);
    for j = 1:points2_detected.Count
        

        coordinates_initial_2=points2_detected(i).Location;
        coordinates_initial_2(3)=1;
        coordinates_initial_2=transpose(coordinates_initial_2);


        coordinates_mimic_2=for_inliers*coordinates_initial_1;
        coordinates_mimic_1=for_inliers\coordinates_initial_2;

        curr_distance=sqrt((coordinates_initial_1(1, 1)-coordinates_mimic_1(1,1)/coordinates_mimic_1(3, 1))^2 ...
            +(coordinates_initial_1(2, 1)-coordinates_mimic_1(2,1)/coordinates_mimic_1(3, 1))^2)+...
                sqrt((coordinates_initial_2(1, 1)-coordinates_mimic_2(1,1)/coordinates_mimic_2(3, 1))^2 ...
            +(coordinates_initial_2(2, 1)-coordinates_mimic_2(2,1)/coordinates_mimic_2(3, 1))^2);


        if (curr_distance<threshold_for_matching && curr_distance<minimum)
            minimum=curr_distance;
            guided_matching_1(k, 1)=coordinates_initial_1(1, 1);
            guided_matching_1(k, 2)=coordinates_initial_1(2, 1);
            guided_matching_2(k, 1)=coordinates_initial_2(1, 1);
            guided_matching_2(k, 2)=coordinates_initial_2(2, 1);
            disp('smaller');
            break
        end
    end
    k=k+1;
end


figure 
showMatchedFeatures(image_1, image_2, guided_matching_1, guided_matching_2)
title('Guided Matching');
%}
