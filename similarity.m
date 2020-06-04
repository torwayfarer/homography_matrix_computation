% function to compute similarity transform
function f=similarity(points_locations)
    machine_zero=2.2204e-10;
    
    numb_of_points=size(points_locations, 1);
    
    % compute the mean of x and y
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sum_of_col=sum(points_locations, 1);
    sum_x=sum_of_col(1, 1);
    sum_y=sum_of_col(1, 2);
    
    mean_x=sum_x/numb_of_points;
    mean_y=sum_y/numb_of_points;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % compute the denominator of scale
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    scale_denum=0.0;
    
    for i = 1 : numb_of_points
        scale_denum=scale_denum+sqrt((points_locations(i, 1)-mean_x)^2+(points_locations(i, 2)-mean_y)^2);
    end
     
    if (scale_denum==0.0000)
        scale_denum=machine_zero;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % compute the scale
    scale=sqrt(2)/(scale_denum/numb_of_points);
    
    % compute final similarity matrix
    f=[scale 0 -scale*mean_x; 0 scale -scale*mean_y; 0 0 1];
    
end
